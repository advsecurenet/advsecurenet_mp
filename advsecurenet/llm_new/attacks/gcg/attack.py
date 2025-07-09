import gc
import torch
from tqdm.auto import tqdm

from advsecurenet.llm_new.core.prompts.attack_prompt import AttackPrompt
from advsecurenet.llm_new.core.prompts.prompt_manger import PromptManager
from advsecurenet.llm_new.core.attack_manager.multi_prompt_attack import MultiPromptAttack

class GCGMultiPromptAttack(MultiPromptAttack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, 
             batch_size=1024, 
             topk=256, 
             temp=1.0, 
             allow_non_ascii=True, 
             target_weight=1.0, 
             control_weight=0.1, 
             verbose=False, 
             opt_only=False,
             filter_cand=True):
        
        opt_only = False  # not yet supported
        main_device = self.models[0].device
        control_cands = []

        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "grad", worker.model)

        grad = None
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get().to(main_device)
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:
                grad = torch.zeros_like(new_grad)
            if grad.shape != new_grad.shape:
                with torch.no_grad():
                    control_cand = self.prompts[j-1].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
                    control_cands.append(
                        self.get_filtered_cands(j-1, control_cand, filter_cand=filter_cand, curr_control=self.control_str)
                    )
                grad = new_grad
            else:
                grad += new_grad

        with torch.no_grad():
            control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
            control_cands.append(
                self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str)
            )
        del grad, control_cand ; gc.collect()

        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        with torch.no_grad():
            for j, cand in enumerate(control_cands):
                progress = tqdm(range(len(self.prompts[0]))) if verbose else range(len(self.prompts[0]))
                for i in progress:
                    for k, worker in enumerate(self.workers):
                        worker(self.prompts[k][i], "logits", worker.model, cand, return_ids=True)
                    logits, ids = zip(*[worker.results.get() for worker in self.workers])
                    loss[j*batch_size:(j+1)*batch_size] += sum([
                        target_weight * self.prompts[k][i].target_loss(logit, id).mean(dim=-1).to(main_device) 
                        for k, (logit, id) in enumerate(zip(logits, ids))
                    ])
                    if control_weight != 0:
                        loss[j*batch_size:(j+1)*batch_size] += sum([
                            control_weight * self.prompts[k][i].control_loss(logit, id).mean(dim=-1).to(main_device) 
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])
                    del logits, ids ; gc.collect()
                    if verbose:
                        current_loss = loss[j * batch_size: (j+1) * batch_size].min().item()/(i+1)
                        progress.set_description(f"loss={current_loss:.4f}")    

            min_idx = loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control = control_cands[model_idx][batch_idx]
            cand_loss = loss[min_idx]

        del control_cands, loss ; gc.collect()
        print("Generated length:", len(self.workers[0].tokenizer(next_control).input_ids[1:]))
        print("Candidate control string:\n", next_control)
        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)
