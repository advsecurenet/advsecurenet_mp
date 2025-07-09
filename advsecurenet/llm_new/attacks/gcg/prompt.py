import numpy as np
import torch

from advsecurenet.llm_new.core.prompts.attack_prompt import AttackPrompt
from advsecurenet.llm_new.core.prompts.prompt_manger import PromptManager
from advsecurenet.llm_new.attacks.gcg.gradients import token_gradients


class GCGAttackPrompt(AttackPrompt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def grad(self, model):
        return token_gradients(
            model,
            self.input_ids.to(model.device),
            self._control_slice,
            self._target_slice,
            self._loss_slice
        )


class GCGPromptManager(PromptManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_control(self, grad, batch_size, topk=256, temp=1.0, allow_non_ascii=True):
        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = float("inf")

        top_indices = (-grad).topk(topk, dim=1).indices
        control_toks = self.control_toks.to(grad.device)
        original_control_toks = control_toks.repeat(batch_size, 1)

        new_token_pos = torch.arange(
            0,
            len(control_toks),
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)

        new_token_val = torch.gather(
            top_indices[new_token_pos], 1,
            torch.randint(0, topk, (batch_size, 1), device=grad.device)
        )

        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_control_toks
