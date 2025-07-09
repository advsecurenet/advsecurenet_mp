import gc
import json
import math
import random
import time
from copy import deepcopy

import numpy as np
import torch


class MultiPromptAttack:
    """Manager for running prompt-based adversarial attacks across multiple models."""

    def __init__(
        self,
        goals,
        targets,
        workers,
        control_init,
        managers,
        test_prefixes=None,
        logfile=None,
        test_goals=None,
        test_targets=None,
        test_workers=None,
    ):
        self.goals = goals
        self.targets = targets
        self.workers = workers
        self.test_workers = test_workers or []
        self.test_goals = test_goals or []
        self.test_targets = test_targets or []
        self.test_prefixes = test_prefixes or []

        self.models = [worker.model for worker in workers]
        self.logfile = logfile
        self.managers = managers

        # Create PromptManager instances for each worker
        self.prompts = [
            managers['PM'](
                goals, targets,
                worker.tokenizer, worker.conv_template,
                control_init, self.test_prefixes, managers
            )
            for worker in workers
        ]

    @property
    def control_str(self):
        return self.prompts[0].control_str

    @control_str.setter
    def control_str(self, control):
        for prompt in self.prompts:
            prompt.control_str = control

    def step(self, *args, **kwargs):
        raise NotImplementedError("Must implement step method in a subclass.")

    def run(
        self,
        n_steps=100,
        batch_size=1024,
        topk=256,
        temp=1,
        allow_non_ascii=True,
        target_weight=1.0,
        control_weight=0.1,
        anneal=True,
        anneal_from=0,
        stop_on_success=True,
        test_steps=50,
        filter_cand=True,
        verbose=True,
    ):
        def should_accept(new_loss, old_loss, step):
            T = max(1 - float(step + 1) / (n_steps + anneal_from), 1e-7)
            return new_loss < old_loss or math.exp(-(new_loss - old_loss) / T) >= random.random()

        loss = best_loss = 1e6
        prev_loss = float("inf")
        best_control = self.control_str
        steps = 0

        for i in range(n_steps):
            if stop_on_success and self.all_models_pass():
                break

            steps += 1
            torch.cuda.empty_cache()
            start = time.time()

            control, loss = self.step(
                batch_size=batch_size,
                topk=topk,
                temp=temp,
                allow_non_ascii=allow_non_ascii,
                target_weight=target_weight,
                control_weight=control_weight,
                filter_cand=filter_cand,
                verbose=verbose,
            )

            runtime = time.time() - start
            if not anneal or should_accept(loss, prev_loss, i + anneal_from):
                self.control_str = control

            prev_loss = loss
            if loss < best_loss:
                best_loss = loss
                best_control = control

            print(f"[Step {i+1}] Loss: {loss:.4f} | Best: {best_loss:.4f}")

        return best_control, best_loss, steps

    def all_models_pass(self):
        jb_results, _, _ = self.test(self.workers, self.prompts)
        return all(all(results) for results in jb_results)

    def test(self, workers, prompts, include_loss=False):
        for j, worker in enumerate(workers):
            worker(prompts[j], "test", worker.model)
        jb_results = [worker.results.get()[0] for worker in workers]
        mb_results = [worker.results.get()[1] for worker in workers]
        loss_results = []

        if include_loss:
            for j, worker in enumerate(workers):
                worker(prompts[j], "test_loss", worker.model)
            loss_results = [worker.results.get() for worker in workers]

        return jb_results, mb_results, loss_results

    def get_filtered_cands(self, worker_index, control_cand, filter_cand=True, curr_control=None):
        cands = []
        worker = self.workers[worker_index]
        for i in range(control_cand.shape[0]):
            decoded_str = worker.tokenizer.decode(control_cand[i], skip_special_tokens=True)
            if not filter_cand or (
                decoded_str != curr_control and
                len(worker.tokenizer(decoded_str, add_special_tokens=False).input_ids) == len(control_cand[i])
            ):
                cands.append(decoded_str)
        return cands or [worker.tokenizer.decode(control_cand[0], skip_special_tokens=True)]
