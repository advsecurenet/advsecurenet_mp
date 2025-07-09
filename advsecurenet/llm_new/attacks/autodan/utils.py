
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Dict


def truthful_decode(tokenizer, ids):  # bad case otherwise: token='▁!', output = '!'
    """
    Examples: <0x0A> -> '\n'; apple -> 'apple', ▁! -> ' !'
    """
    dummy_id = tokenizer.convert_tokens_to_ids('a')
    if isinstance(ids, list):
        text = tokenizer.decode([dummy_id] + ids + [dummy_id], skip_special_tokens=False).strip()[
               1:-1]  # it automatically adds a space in the front when index is <s> or </s>, so strip it out
    elif isinstance(ids, torch.Tensor) and len(ids.shape) == 1:
        dummy_id = torch.tensor([dummy_id], dtype=torch.int64).to(ids.device)
        text = tokenizer.decode(torch.cat([dummy_id, ids, dummy_id]),
                                skip_special_tokens=False).strip()[1:-1]
    else:
        raise NotImplementedError(f"Unexpected type {type(ids)} of ids {ids}")
    return text


class VocabularyTracker:
    def __init__(self, vocab_size: int, N: int = 4, device='cuda'):
        self.vocab_size = vocab_size
        # Initialize the table with zeros. Columns represent:
        # Explored (0 or 1), Target Loss, Control Loss, Total Loss
        self.device = device
        self.table = torch.zeros(vocab_size, N, device=device)
        # Separate tensors for ctrl and ctrl_last_tok information
        self.ctrl: Dict[int, Tensor] = {}  # Maps token index to a 1-D ctrl tensor
        # Indices for easy reference
        self.EXPLORED_IDX = 0
        self.TARGET_LOSS_IDX = 1
        self.CONTROL_LOSS_IDX = 2
        self.TOTAL_LOSS_IDX = 3

    def update(self,
               token_indices: Tensor,
               target_losses: Tensor,
               control_losses: Tensor,
               losses: Tensor,
               ctrls: Tensor):
        """
        Update the records for given tokens.

        Args:
            token_indices (Tensor): A 1-D tensor of token indices (int64).
            target_losses (Tensor): A 1-D tensor of target losses (float32 or float64).
            control_losses (Tensor): A 1-D tensor of control losses (float32 or float64).
            losses (Tensor): A 1-D tensor of combined losses.
            ctrls (Tensor): A 2-D tensor of shape (N, k) for "ctrl".
        """
        assert token_indices.dim() == 1, "token_indices must be a 1-D tensor"
        assert target_losses.dim() == 1, "target_losses must be a 1-D tensor"
        assert control_losses.dim() == 1, "control_losses must be a 1-D tensor"
        assert losses.dim() == 1, "control_losses must be a 1-D tensor"
        assert ctrls.dim() == 2, "ctrls must be a 2-D tensor with shape (N, k)"

        # Mark tokens as explored and update losses
        self.table[token_indices, self.EXPLORED_IDX] = 1
        self.table[token_indices, self.TARGET_LOSS_IDX] = target_losses
        self.table[token_indices, self.CONTROL_LOSS_IDX] = control_losses
        self.table[token_indices, self.TOTAL_LOSS_IDX] = losses

        # Update ctrl and ctrl_last_tok information
        for i, idx in enumerate(token_indices):
            self.ctrl[idx.item()] = ctrls[i]

    def reset(self):
        self.table = torch.zeros(self.vocab_size, 4, device=self.device)
        self.ctrl = {}

    def sample(self, topk: int, temperature: float, eps: float = 1e-10) -> (Tensor, Tensor, Tensor):
        explored_tokens = self.table[self.table[:, self.EXPLORED_IDX] == 1]
        if explored_tokens.size(0) == 0:
            return -1, torch.tensor([]), torch.tensor([])  # Handle case with no explored tokens

        # Rank tokens based on total loss and select top-k
        losses = explored_tokens[:, self.TOTAL_LOSS_IDX]
        _, indices = torch.topk(losses, topk, largest=False)
        topk_losses = losses[indices]

        # Convert losses to probabilities
        probabilities = F.softmax((-topk_losses) / (temperature + eps), dim=0)

        # Sample one token based on the calculated probabilities
        sampled_index = torch.multinomial(probabilities, 1).item()
        original_indices = torch.nonzero(self.table[:, self.EXPLORED_IDX] == 1, as_tuple=True)
        # Since original_indices is a tuple of tensors, directly use the first tensor which contains the indices we need
        sampled_token_index = original_indices[0][indices][sampled_index].item()

        # Retrieve the ctrl, ctrl_last_tok, and losses for the sampled token
        sampled_ctrl = self.ctrl.get(sampled_token_index, torch.tensor([]))
        sampled_target_loss = self.table[sampled_token_index, self.TARGET_LOSS_IDX].item()
        sampled_control_loss = self.table[sampled_token_index, self.CONTROL_LOSS_IDX].item()
        sampled_total_loss = self.table[sampled_token_index, self.TOTAL_LOSS_IDX].item()

        return (sampled_token_index, sampled_ctrl,
                sampled_target_loss, sampled_control_loss, sampled_total_loss)

    def get_explored_ids(self) -> Tensor:
        """
        Returns the token indices of the explored tokens.
        """
        explored_mask = self.table[:, self.EXPLORED_IDX] == 1
        explored_indices = torch.nonzero(explored_mask, as_tuple=True)[0]
        return explored_indices

    def get_top_k_tokens(self, topk: int, temperature: float, eps: float = 1e-10) -> Tuple[
        Tensor, Tensor]:
        """
        Returns the top-k ranked tokens based on their losses and their probabilities.
        For debugging only

        Args:
            topk (int): The number of top tokens to return.
            temperature (float): The temperature parameter for the softmax function.
            eps (float): A small value to avoid division by zero in softmax.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing two tensors. The first tensor contains the indices of the top-k tokens.
                                   The second tensor contains the corresponding probabilities of these tokens.
        """
        explored_tokens = self.table[self.table[:, self.EXPLORED_IDX] == 1]
        if explored_tokens.size(0) == 0:
            return torch.tensor([]), torch.tensor([])  # Handle case with no explored tokens

        # Use total loss for ranking
        total_losses = explored_tokens[:, self.TOTAL_LOSS_IDX]
        # Convert losses to probabilities
        probabilities = F.softmax((-total_losses) / (temperature + eps), dim=0)

        # Rank tokens based on probabilities and select top-k
        topk_probabilities, topk_indices = torch.topk(probabilities, topk, largest=True)

        # Extract the original token indices for the top-k tokens
        original_indices = torch.nonzero(self.table[:, self.EXPLORED_IDX] == 1, as_tuple=True)[0]
        topk_token_indices = original_indices[topk_indices]

        return topk_token_indices, topk_probabilities


tests = {
    'vicuna_v1.1-tail':
        [
            {
                'control': ' !',
                'control_last_tok': '▁!',
                'assistant_role': 'ASSISTANT:',
                'strings_before': "<s> A chat between a curious user and an artificial intelligence assistant. "
                                  "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                                  "USER:{goal}",
                'strings_after': "{suffix} ASSISTANT: {target}",
                'strings_all': "<s> A chat between a curious user and an artificial intelligence assistant. "
                               "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                               "USER:{goal} !{suffix} ASSISTANT: {target}",
            },
            {
                'control': ' Question:',
                'control_last_tok': ':',
                'assistant_role': 'ASSISTANT:',
                'strings_before': "<s> A chat between a curious user and an artificial intelligence assistant. "
                                  "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                                  "USER:{goal} Question",
                'strings_after': "{suffix} ASSISTANT: {target}",
                'strings_all': "<s> A chat between a curious user and an artificial intelligence assistant. "
                               "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                               "USER:{goal} Question:{suffix} ASSISTANT: {target}",
            },
            {
                'control': ' At the end\n',
                'control_last_tok': '<0x0A>',
                'assistant_role': 'ASSISTANT:',
                'strings_before': "<s> A chat between a curious user and an artificial intelligence assistant. "
                                  "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                                  "USER:{goal} At the end",
                'strings_after': "{suffix} ASSISTANT: {target}",
                'strings_all': "<s> A chat between a curious user and an artificial intelligence assistant. "
                               "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                               "USER:{goal} At the end\n{suffix} ASSISTANT: {target}",
            },
            {
                'control': ', using a',
                'control_last_tok': '▁a',
                'assistant_role': 'ASSISTANT:',
                'strings_before': "<s> A chat between a curious user and an artificial intelligence assistant. "
                                  "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                                  "USER:{goal}, using",
                'strings_after': "{suffix} ASSISTANT: {target}",
                'strings_all': "<s> A chat between a curious user and an artificial intelligence assistant. "
                               "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                               "USER:{goal}, using a{suffix} ASSISTANT: {target}",
            },
        ],
    'vicuna_v1.1-head':
        [
            {
                'control': ' !',
                'control_last_tok': '▁!',
                'assistant_role': 'ASSISTANT:',
                'strings_before': "<s> A chat between a curious user and an artificial intelligence assistant. "
                                  "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                                  "USER:",
                'strings_after': "{suffix}{goal} ASSISTANT: {target}",
                'strings_all': "<s> A chat between a curious user and an artificial intelligence assistant. "
                               "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                               "USER: !{suffix}{goal} ASSISTANT: {target}",
            },
            {
                'control': ' Question:',
                'control_last_tok': ':',
                'assistant_role': 'ASSISTANT:',
                'strings_before': "<s> A chat between a curious user and an artificial intelligence assistant. "
                                  "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                                  "USER: Question",
                'strings_after': "{suffix}{goal} ASSISTANT: {target}",
                'strings_all': "<s> A chat between a curious user and an artificial intelligence assistant. "
                               "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                               "USER: Question:{suffix}{goal} ASSISTANT: {target}",
            },
            {
                'control': ' At the end\n',
                'control_last_tok': '<0x0A>',
                'assistant_role': 'ASSISTANT:',
                'strings_before': "<s> A chat between a curious user and an artificial intelligence assistant. "
                                  "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                                  "USER: At the end",
                'strings_after': "{suffix}{goal} ASSISTANT: {target}",
                'strings_all': "<s> A chat between a curious user and an artificial intelligence assistant. "
                               "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                               "USER: At the end\n{suffix}{goal} ASSISTANT: {target}",
            },
            {
                'control': ', using a',
                'control_last_tok': '▁a',
                'assistant_role': 'ASSISTANT:',
                'strings_before': "<s> A chat between a curious user and an artificial intelligence assistant. "
                                  "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                                  "USER:, using",
                'strings_after': "{suffix}{goal} ASSISTANT: {target}",
                'strings_all': "<s> A chat between a curious user and an artificial intelligence assistant. "
                               "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                               "USER:, using a{suffix}{goal} ASSISTANT: {target}",
            },
        ],
    'llama-2-tail':
        [
            {
                'control': ' !',
                'control_last_tok': '▁!',
                'assistant_role': '[/INST]',
                'strings_before': "<s> [INST]{goal}",
                'strings_after': "{suffix} [/INST] {target}",
                'strings_all': "<s> [INST]{goal} !{suffix} [/INST] {target}",
            },
            {
                'control': ' Question:',
                'control_last_tok': ':',
                'assistant_role': '[/INST]',
                'strings_before': "<s> [INST]{goal} Question",
                'strings_after': "{suffix} [/INST] {target}",
                'strings_all': "<s> [INST]{goal} Question:{suffix} [/INST] {target}",
            },
            {
                'control': ' At the end\n',
                'control_last_tok': '<0x0A>',
                'assistant_role': '[/INST]',
                'strings_before': "<s> [INST]{goal} At the end",
                'strings_after': "{suffix} [/INST] {target}",
                'strings_all': "<s> [INST]{goal} At the end\n{suffix} [/INST] {target}",
                # todo: whether to add the space before [/INST]
            },
            {
                'control': ', using a',
                'control_last_tok': '▁a',
                'assistant_role': '[/INST]',
                'strings_before': "<s> [INST]{goal}, using",
                'strings_after': "{suffix} [/INST] {target}",
                'strings_all': "<s> [INST]{goal}, using a{suffix} [/INST] {target}"
            },
        ],
    'llama-2-head':
        [
            {
                'control': ' !',
                'control_last_tok': '▁!',
                'assistant_role': '[/INST]',
                'strings_before': "<s> [INST]",
                'strings_after': "{suffix}{goal} [/INST] {target}",
                'strings_all': "<s> [INST] !{suffix}{goal} [/INST] {target}",
            },
            {
                'control': ' Question:',
                'control_last_tok': ':',
                'assistant_role': '[/INST]',
                'strings_before': "<s> [INST] Question",
                'strings_after': "{suffix}{goal} [/INST] {target}",
                'strings_all': "<s> [INST] Question:{suffix}{goal} [/INST] {target}",
            },
            {
                'control': ' At the end\n',
                'control_last_tok': '<0x0A>',
                'assistant_role': '[/INST]',
                'strings_before': "<s> [INST] At the end",
                # todo: whether to add the space before At
                'strings_after': "{suffix}{goal} [/INST] {target}",
                # todo: forcibally added a '.' after goal, does not apply to customized loss with other endings
                'strings_all': "<s> [INST] At the end\n{suffix}{goal} [/INST] {target}",
            },
            # fixme: this will give an error b/c of sticky tokenization: "],"
            # {
            #     'control': ', using a',
            #     'control_last_tok': '▁a',
            #     'assistant_role': '[/INST]',
            #     'strings_before': "<s> [INST], using",
            #     'strings_after': "{suffix}{goal} [/INST] {target}",
            #     'strings_all': "<s> [INST], using a{suffix}{goal} [/INST] {target}",
            # },
        ],
    'zero_shot-tail':
        [
            {
                'control': ' !',
                'control_last_tok': '▁!',
                'assistant_role': '### Assistant:',
                'strings_before': "<s>A chat between a curious human and an artificial intelligence assistant. "
                                  "The assistant gives helpful, detailed, and polite answers to the human's questions."
                                  "\n### Human:{goal}",
                'strings_after': "{suffix}\n### Assistant: {target}",
                'strings_all': "<s>A chat between a curious human and an artificial intelligence assistant. "
                               "The assistant gives helpful, detailed, and polite answers to the human's questions."
                               "\n### Human:{goal} !{suffix}"
                               "\n### Assistant: {target}",
            },
            {
                'control': ' Question:',
                'control_last_tok': ':',
                'assistant_role': '### Assistant:',
                'strings_before': "<s>A chat between a curious human and an artificial intelligence assistant. "
                                  "The assistant gives helpful, detailed, and polite answers to the human's questions."
                                  "\n### Human:{goal} Question",
                'strings_after': "{suffix}\n### Assistant: {target}",
                'strings_all': "<s>A chat between a curious human and an artificial intelligence assistant. "
                               "The assistant gives helpful, detailed, and polite answers to the human's questions."
                               "\n### Human:{goal} Question:{suffix}"
                               "\n### Assistant: {target}",
            },
            {
                'control': ' At the end\n',
                'control_last_tok': '<0x0A>',
                'assistant_role': '### Assistant:',
                'strings_before': "<s>A chat between a curious human and an artificial intelligence assistant. "
                                  "The assistant gives helpful, detailed, and polite answers to the human's questions."
                                  "\n### Human:{goal} At the end",
                'strings_after': "{suffix}\n### Assistant: {target}",
                'strings_all': "<s>A chat between a curious human and an artificial intelligence assistant. "
                               "The assistant gives helpful, detailed, and polite answers to the human's questions."
                               "\n### Human:{goal} At the end\n{suffix}"
                               "\n### Assistant: {target}",
            },
            {
                'control': ', using a',
                'control_last_tok': '▁a',
                'assistant_role': '### Assistant:',
                'strings_before': "<s>A chat between a curious human and an artificial intelligence assistant. "
                                  "The assistant gives helpful, detailed, and polite answers to the human's questions."
                                  "\n### Human:{goal}, using",
                'strings_after': "{suffix}\n### Assistant: {target}",
                'strings_all': "<s>A chat between a curious human and an artificial intelligence assistant. "
                               "The assistant gives helpful, detailed, and polite answers to the human's questions."
                               "\n### Human:{goal}, using a{suffix}"
                               "\n### Assistant: {target}",
            },
        ],
    'zero_shot-head':
        [
            {
                'control': ' !',
                'control_last_tok': '▁!',
                'assistant_role': '### Assistant:',
                'strings_before': "<s>A chat between a curious human and an artificial intelligence assistant. "
                                  "The assistant gives helpful, detailed, and polite answers to the human's questions."
                                  "\n### Human:",
                'strings_after': "{suffix}{goal}\n### Assistant: {target}",
                'strings_all': "<s>A chat between a curious human and an artificial intelligence assistant. "
                               "The assistant gives helpful, detailed, and polite answers to the human's questions."
                               "\n### Human: !{suffix}{goal}"
                               "\n### Assistant: {target}",
            },
            {
                'control': ' Question:',
                'control_last_tok': ':',
                'assistant_role': '### Assistant:',
                'strings_before': "<s>A chat between a curious human and an artificial intelligence assistant. "
                                  "The assistant gives helpful, detailed, and polite answers to the human's questions."
                                  "\n### Human: Question",
                'strings_after': "{suffix}{goal}\n### Assistant: {target}",
                'strings_all': "<s>A chat between a curious human and an artificial intelligence assistant. "
                               "The assistant gives helpful, detailed, and polite answers to the human's questions."
                               "\n### Human: Question:{suffix}{goal}"
                               "\n### Assistant: {target}",
            },
            {
                'control': ' At the end\n',
                'control_last_tok': '<0x0A>',
                'assistant_role': '### Assistant:',
                'strings_before': "<s>A chat between a curious human and an artificial intelligence assistant. "
                                  "The assistant gives helpful, detailed, and polite answers to the human's questions."
                                  "\n### Human: At the end",
                'strings_after': "{suffix}{goal}\n### Assistant: {target}",
                'strings_all': "<s>A chat between a curious human and an artificial intelligence assistant. "
                               "The assistant gives helpful, detailed, and polite answers to the human's questions."
                               "\n### Human: At the end\n{suffix}{goal}"
                               "\n### Assistant: {target}",
            },
            {
                'control': ', using a',
                'control_last_tok': '▁a',
                'assistant_role': '### Assistant:',
                'strings_before': "<s>A chat between a curious human and an artificial intelligence assistant. "
                                  "The assistant gives helpful, detailed, and polite answers to the human's questions."
                                  "\n### Human:, using",
                'strings_after': "{suffix}{goal}\n### Assistant: {target}",
                'strings_all': "<s>A chat between a curious human and an artificial intelligence assistant. "
                               "The assistant gives helpful, detailed, and polite answers to the human's questions."
                               "\n### Human:, using a{suffix}{goal}"
                               "\n### Assistant: {target}",
            },
        ],
    'oasst_pythia-tail':
        [
            {
                'control': ' :',
                'control_last_tok': 'Ġ:',
                'assistant_role': '<|assistant|>',
                'strings_before': "<|prompter|>{goal}",
                'strings_after': "{suffix}<|endoftext|><|assistant|>{target}",
                'strings_all': "<|prompter|>{goal} :{suffix}<|endoftext|><|assistant|>{target}",
            },
            {
                'control': ' Question:',
                'control_last_tok': ':',
                'assistant_role': '<|assistant|>',
                'strings_before': "<|prompter|>{goal} Question",
                'strings_after': "{suffix}<|endoftext|><|assistant|>{target}",
                'strings_all': "<|prompter|>{goal} Question:{suffix}<|endoftext|><|assistant|>{target}",
            },
            # {
            #     'control': ' At the end\n',
            #     'control_last_tok': '<0x0A>',
            #     'assistant_role': '<|assistant|>',
            #     'strings_before': "<|prompter|>{goal} At the end",
            #     'strings_after': " {suffix}<|endoftext|><|assistant|>{target}",
            #     'strings_all': "<|prompter|>{goal} At the end{suffix}<|endoftext|><|assistant|>{target}",
            # },
            {
                'control': ', using a',
                'control_last_tok': 'Ġa',
                'assistant_role': '<|assistant|>',
                'strings_before': "<|prompter|>{goal}, using",
                'strings_after': "{suffix}<|endoftext|><|assistant|>{target}",
                'strings_all': "<|prompter|>{goal}, using a{suffix}<|endoftext|><|assistant|>{target}",
            },
        ],
    'oasst_pythia-head':
        [
            {
                'control': ' :',
                'control_last_tok': 'Ġ:',
                'assistant_role': '<|assistant|>',
                'strings_before': "<|prompter|>",
                'strings_after': "{suffix}{goal}<|endoftext|><|assistant|>{target}",
                'strings_all': "<|prompter|> :{suffix}{goal}<|endoftext|><|assistant|>{target}",
            },
            {
                'control': ' Question:',
                'control_last_tok': ':',
                'assistant_role': '<|assistant|>',
                'strings_before': "<|prompter|> Question",
                'strings_after': "{suffix}{goal}<|endoftext|><|assistant|>{target}",
                'strings_all': "<|prompter|> Question:{suffix}{goal}<|endoftext|><|assistant|>{target}",
            },
            # {
            #     'control': ' At the end\n',
            #     'control_last_tok': '<0x0A>',
            #     'assistant_role': '<|assistant|>',
            #     'strings_before': "<|prompter|> At the end",
            #     'strings_after': "{suffix} {goal}<|endoftext|><|assistant|>{target}",
            #     'strings_all': "<|prompter|> At the end{suffix} {goal}<|endoftext|><|assistant|>{target}",
            # },
            {
                'control': ', using a',
                'control_last_tok': 'Ġa',
                'assistant_role': '<|assistant|>',
                'strings_before': "<|prompter|>, using",
                'strings_after': "{suffix}{goal}<|endoftext|><|assistant|>{target}",
                'strings_all': "<|prompter|>, using a{suffix}{goal}<|endoftext|><|assistant|>{target}",
            },
        ],
    'mistral-tail':
        [
            {
                'control': ' !',
                'control_last_tok': '▁!',
                'assistant_role': '[/INST]',
                'strings_before': "<s>[INST]{goal}",
                'strings_after': "{suffix} [/INST] {target}",
                'strings_all': "<s>[INST]{goal} !{suffix} [/INST] {target}",
            },
            {
                'control': ' Question:',
                'control_last_tok': ':',
                'assistant_role': '[/INST]',
                'strings_before': "<s>[INST]{goal} Question",
                'strings_after': "{suffix} [/INST] {target}",
                'strings_all': "<s>[INST]{goal} Question:{suffix} [/INST] {target}",
            },
            {
                'control': ' At the end\n',
                'control_last_tok': '<0x0A>',
                'assistant_role': '[/INST]',
                'strings_before': "<s>[INST]{goal} At the end",
                'strings_after': "{suffix} [/INST] {target}",
                'strings_all': "<s>[INST]{goal} At the end\n{suffix} [/INST] {target}",
            },
            {
                'control': ', using a',
                'control_last_tok': '▁a',
                'assistant_role': '[/INST]',
                'strings_before': "<s>[INST]{goal}, using",
                'strings_after': "{suffix} [/INST] {target}",
                'strings_all': "<s>[INST]{goal}, using a{suffix} [/INST] {target}"
            },
        ],
    'mistral-head':
        [
            {
                'control': ' !',
                'control_last_tok': '▁!',
                'assistant_role': '[/INST]',
                'strings_before': "<s>[INST]",
                'strings_after': "{suffix}{goal} [/INST] {target}",
                'strings_all': "<s>[INST] !{suffix}{goal} [/INST] {target}",
            },
            {
                'control': ' Question:',
                'control_last_tok': ':',
                'assistant_role': '[/INST]',
                'strings_before': "<s>[INST] Question",
                'strings_after': "{suffix}{goal} [/INST] {target}",
                'strings_all': "<s>[INST] Question:{suffix}{goal} [/INST] {target}",
            },
            {
                'control': ' At the end\n',
                'control_last_tok': '<0x0A>',
                'assistant_role': '[/INST]',
                'strings_before': "<s>[INST] At the end",
                # todo: whether to add the space before At
                'strings_after': "{suffix}{goal} [/INST] {target}",
                # todo: forcibally added a '.' after goal, does not apply to customized loss with other endings
                'strings_all': "<s>[INST] At the end\n{suffix}{goal} [/INST] {target}",
            },
        ],
}


def run_test(attack_prompt, suffix):
    backup_control = attack_prompt.control
    backup_control_last_tok = attack_prompt.control_last_tok
    template_name = attack_prompt.conv_template.name
    # begin test
    cases = tests[template_name + '-' + attack_prompt.insert_loc]
    goal = attack_prompt.goal
    if template_name in ['vicuna_v1.1', 'llama-2', 'zero_shot', 'mistral']:
        if attack_prompt.goal and attack_prompt.insert_loc == 'tail':
            goal = ' ' + attack_prompt.goal  # to handle the empty goal case

    tokenizer = attack_prompt.tokenizer
    for case in cases:
        attack_prompt.control_last_tok = case['control_last_tok']
        attack_prompt.control = case['control']
        strings_before = case['strings_before'].format(goal=goal,
                                                       target=attack_prompt.target,
                                                       suffix=suffix)
        strings_after = case['strings_after'].format(goal=goal,
                                                     target=attack_prompt.target,
                                                     suffix=suffix)
        if template_name in ['vicuna_v1.1', 'llama-2', 'zero_shot', 'mistral']:
            strings_before = strings_before.rstrip(' ')
            strings_after = strings_after.lstrip(' ')
        strings_all = case['strings_all'].format(goal=goal,
                                                 target=attack_prompt.target,
                                                 suffix=suffix)
        attack_prompt._update_ids_individually()
        ids = attack_prompt.input_ids

        check_list = [
            ['control',
             case['control'],
             truthful_decode(tokenizer, ids[attack_prompt._control_slice]),
             ],
            ['last control tok',
             tokenizer.convert_tokens_to_ids(case['control_last_tok']),
             ids[attack_prompt._control_last_tok_slice],
             ],
            ['target',
             attack_prompt.target,
             tokenizer.decode(ids[attack_prompt._target_slice]),
             ],
            ['strings_before',
             strings_before,
             tokenizer.decode(ids[:attack_prompt._control_last_tok_slice.start]),
             ],
            ['strings_after',
             strings_after,
             tokenizer.decode(ids[attack_prompt._control_last_tok_slice.stop:]),
             ],
            ['strings_all',
             strings_all,
             tokenizer.decode(ids),
             ],
            ['assistant_role',
             case['assistant_role'][-1],  # only need assistant_role.stop to be correct
             tokenizer.decode(ids[attack_prompt._assistant_role_slice])[-1],
             ],
        ]
        for check_name, expected_output, actual_output in check_list:
            assert expected_output == actual_output, \
                (f"Mismatched {check_name}!\n"
                 f"Expected: \n{expected_output}\n"
                 f"Actual: \n{actual_output}\n"
                 f"Current control {check_list[0][1]}")

    print(f"All unit tests passed ({len(cases)}/{len(cases)})")
    # recover original settings after test
    attack_prompt.control = backup_control
    attack_prompt.control_last_tok = backup_control_last_tok
    attack_prompt._update_ids_individually()
