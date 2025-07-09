import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


from advsecurenet.llm_new.core.prompts.attack_prompt import AttackPrompt
from advsecurenet.llm_new.core.prompts.prompt_manger import PromptManager
from advsecurenet.llm_new.core.attack_manager.multi_prompt_attack import MultiPromptAttack
from advsecurenet.llm_new.core.models.embeddings import get_embedding_matrix, get_embeddings
from advsecurenet.llm_new.core.attack_manager.attack_manager import token_to_word
from advsecurenet.llm_new.attacks.autodan.utils import  truthful_decode, VocabularyTracker

class AutoDANAttackPrompt(AttackPrompt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preliminary_select(self, model):
        # harmfulness objective
        assert str(self.input_ids.device) == 'cpu', f"input_ids is in {self.input_ids.device}"
        input_ids = self.input_ids.to(model.device)
        embed_weights = get_embedding_matrix(model)

        one_hot = torch.zeros(
            input_ids[self._control_last_tok_slice].shape[0],  # only need grad for last tok
            embed_weights.shape[0],
            device=model.device,
            dtype=embed_weights.dtype
        )
        one_hot.scatter_(
            1,
            input_ids[self._control_last_tok_slice].unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
        )
        one_hot.requires_grad_()
        input_embeds = (one_hot @ embed_weights).unsqueeze(0)

        # now stitch it together with the rest of the embeddings
        embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
        full_embeds = torch.cat(
            [
                embeds[:, :self._control_last_tok_slice.start, :],
                input_embeds,
                embeds[:, self._control_last_tok_slice.stop:, :]
            ],
            dim=1)

        logits = model(inputs_embeds=full_embeds).logits
        targets = input_ids[self._target_slice]
        loss = F.cross_entropy(logits[0, self._loss_slice, :], targets)
        # loss = -(-loss).exp()  # geometric avg
        loss.backward()
        target_grad = - one_hot.grad.clone()  # larger the better

        # readability objective
        with torch.no_grad():
            ctrl_context_ids = self.ctrl_context_ids.unsqueeze(0).to(model.device)
            ctrl_context_logits = model(input_ids=ctrl_context_ids).logits
            # print(self.tokenizer.batch_decode(ctrl_context_ids))  # for sanity check
        last_tok_ll = ctrl_context_logits[:, -1, :].log_softmax(dim=1)
        # del input_ids, ctrl_context_logits; gc.collect()  # doesn't seem to have notable effect
        return target_grad, last_tok_ll  # both of shape (1, vocab_size)


class AutoDANPromptManager(PromptManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def find_dynamic_weight(self, target, control, p, topk, tol_p=1e-2, tol=1e-5):
        control_prob = control.softmax(0)
        total_sum_control = control_prob.sum()
        required_sum = p * total_sum_control

        low, high = 0, 1e5  # initial range for dynamic weight
        dynamic_weight = (low + high) / 2

        while high - low > tol:
            combined = control + dynamic_weight * target
            _, indices = torch.topk(combined, topk)
            top_sum_control = control_prob[indices].sum()

            if abs(top_sum_control - required_sum) < tol_p:
                break

            if top_sum_control < required_sum:
                high = dynamic_weight
            else:
                low = dynamic_weight

            dynamic_weight = (low + high) / 2

        combined = control + dynamic_weight * target
        _, indices = torch.topk(combined, topk)
        top_sum_control = control_prob[indices].sum()

        if abs(top_sum_control - required_sum) > tol_p and dynamic_weight > tol:  # avoid inf * negative
            # final adjustment to ensure top_sum_control is at least the required sum
            if top_sum_control < required_sum:
                dynamic_weight -= tol
                combined = control + dynamic_weight * target
                _, indices = torch.topk(combined, topk)
                top_sum_control = control_prob[indices].sum()
            print(f"Cannot find desired weight, exiting with top sum {top_sum_control}. "
                  f"Desired sum {required_sum}")

        addendum = torch.topk(combined, topk + 1).indices[topk]
        return dynamic_weight, indices, addendum

    def sample_control(self, target_grad, last_tok_ll, batch_size, topk=None,
                       allow_non_ascii=True, cur_step=0, w_tar_1=None, explored_ids=[],
                       dynamic_weight=False):
        if topk is None:
            batch_size = topk
            print(f"param 'topk' not provided, setting as batch_size")

        # minus one for adding the greedily selected token while keeping the same batch size
        _batch_size = batch_size - 1
        _topk = topk - 1

        if not allow_non_ascii:
            target_grad[:, self._nonascii_toks.to(target_grad.device)] = -np.infty

        if len(explored_ids) > 0:
            target_grad[:, explored_ids] = -np.infty

        # optional heuristic: ban some tokens to affect the generation results
        if self.tokenizer.bos_token == '<|endoftext|>':  # for pythia
            ban_tokens = self.tokenizer.all_special_ids  # manually ban special ids like sos and eos
        elif self.tokenizer.bos_token == '<s>':  # others llms
            special_tokens = [0, 1, 2]  # manually ban special tokens: unknown, sos, and eos. added during developing, can potentially be simplified
            ban_tokens = self.tokenizer.all_special_ids + special_tokens

            # the following's effect is unknown. added during developing and haven't removed since
            # token_ends_with_space = [259, 268, 308, 418, 462, 539, 632, 795, 965, 1669, 1678, 3986, 4706, 9651, 18884, 29871]
            # ban_tokens = self.tokenizer.all_special_ids + token_ends_with_space + special_tokens  # all_special_ids = [0, 1, 2]

            # optional style preference: disallow linebreak in the first several toks
            disliked_toks = [self.tokenizer.convert_tokens_to_ids("<0x0A>")]  # <0x0A> = \n
            # disliked_toks += [376]  # "▁\"": 376,
            if cur_step < 50:
                ban_tokens += disliked_toks
        else:
            ban_tokens = []
            print(f"Warning: Unknown tokenizer {self.tokenizer}! Unknown tokenization error may occur.")

        # heuristic to prevent connecting the last word of user request with first new tok so that the original meaning may be altered
        # used only for llama2 - individual behavior
        # method: choose toks only from those with whitespace prefixes
        if (self._prompts[0].conv_template.name == 'llama-2'
                and len(self._prompts[0].control_toks) == 1):  # only affect the first tok
            liked_tokens = [tok_id for tok_id, tok in enumerate(self.tokenizer.get_vocab())
                            if tok.startswith('▁')] + [29892, 29889]  # toks with whitespace prefix. add comma and period
            ban_tokens = [i for i in range(self.tokenizer.vocab_size) if i not in liked_tokens]

        target_grad[:, ban_tokens] = -np.infty

        target_obj = target_grad[0]
        control_obj = last_tok_ll[0]

        if dynamic_weight:
            # interpretable (dynamic) weighting method
            dynamic_weight, topk_indices, addendum_tok = self.find_dynamic_weight(
                target_obj, control_obj, p=w_tar_1, topk=_topk)
        else:
            # fixed weighting method
            combined_obj = control_obj + (
                        w_tar_1 + 1e-15) * target_obj  # adding eps to handle 0 * inf = nan
            _, topk_indices = torch.topk(combined_obj, _topk)
            addendum_tok = torch.topk(combined_obj, _topk + 1).indices[_topk]

        top_indices = topk_indices.unsqueeze(0)

        control_toks = self.control_toks.to(target_grad.device)
        original_control_toks = control_toks.repeat(_batch_size, 1)
        new_token_pos = torch.tensor([len(control_toks) - 1] * _batch_size,
                                     device=target_grad.device).type(torch.int64)
        last_token_pos = torch.tensor([0] * _batch_size,
                                      device=target_grad.device).type(torch.int64)
        if _batch_size < top_indices.shape[0] * _topk:  # do sampling
            new_token_val = torch.gather(
                top_indices[last_token_pos], 1,  # new_token_pos
                torch.randint(0, _topk, (_batch_size, 1),
                device=target_grad.device)
            )
        else:  # choose all
            new_token_val = torch.gather(
                top_indices[last_token_pos], 1,  # new_token_pos
                torch.arange(0, _topk, _topk / _batch_size, device=target_grad.device
                             ).type(torch.int64).unsqueeze(1)
            )
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        if not torch.any(new_token_val == control_toks[-1:]):  # greedy construction
            new_control_toks = torch.cat((new_control_toks, control_toks.unsqueeze(0)))
            new_token_val = torch.cat((new_token_val, control_toks[-1:].unsqueeze(0)))
        else:
            addendum = torch.cat((control_toks[:-1], addendum_tok.reshape(1)))
            new_control_toks = torch.cat((new_control_toks, addendum.unsqueeze(0)))
            new_token_val = torch.cat((new_token_val, addendum_tok.reshape(1, 1)))
        return new_control_toks, new_token_val  # append to the last to avoid removing promising candidates when choosing all


class AutoDANMultiPromptAttack(MultiPromptAttack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_tracker = VocabularyTracker(self.workers[0].tokenizer.vocab_size,
                                               device=self.models[0].device)

    def step(self,
             batch_size=1024,
             topk=256,
             ctrl_temp=1,
             allow_non_ascii=True,
             target_weight=1,
             w_tar_1=None,
             w_tar_2=None,
             verbose=False,
             filter_cand=True,
             diversity_coef=None,
             steps=0):

        main_device = self.models[0].device
        tokenizer = self.workers[0].tokenizer  # implicit assumptions for aggregating all grads: all LLMs use the same vocab
        # self.prompts contains one or two (writer + victim) prompt managers
        assert len(self.prompts) <= 2, "Currently supports up to two LLMs (writer + victim)"

        # begin step 1 (preliminary selection)
        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "preliminary_select", worker.model)

        # aggregate gradients from different models and behaviors
        # currently only support one or two (writer + victim) LLMs. Please refer to GCG for ensemble implementation
        pm_victim = self.prompts[0]  # prompt manager of victim LLM

        # get victim LLM's target loss grad and next token's log-likelihood
        target_grad, last_tok_ll, last_tok_lls = self.workers[0].results.get()
        # del target_grad_, last_tok_ll_; gc.collect()  # commented out due to no noticeable effect found

        # (optional) get writer LLM's next token's log-likelihood
        if len(self.workers) > 1:
            _, last_tok_ll = self.workers[1].results.get()
            last_tok_ll = last_tok_ll.to(main_device)  # move from cuda:1 to cuda:0
            # del target_grad_, last_tok_ll_; gc.collect()  # commented out due to no noticeable effect found

        target_grad = target_grad / len(pm_victim)
        last_tok_ll = last_tok_ll / len(pm_victim)
        last_tok_ll = last_tok_ll.log_softmax(dim=1)  # re-normalize after aggregating different behaviors
        with torch.no_grad():
            ctrl_cand_ids, last_tok_cand_ids = pm_victim.sample_control(
                target_grad, last_tok_ll, batch_size, topk, allow_non_ascii, steps, w_tar_1,
                explored_ids=self.vocab_tracker.get_explored_ids(), dynamic_weight=self.dynamic_weight)
            # last_tok_cand_strs = [tokenizer.convert_ids_to_tokens(
            #     last_tok_id) for last_tok_id in last_tok_cand_ids]

        # begin step 2 (fine selection)
        loss_target = torch.zeros(batch_size).to(main_device)
        loss_control = torch.zeros(batch_size).to(main_device)
        with torch.no_grad():
            # Looping through the prompts at this level is less elegant, but
            # we can manage VRAM better this way
            progress = tqdm(range(len(pm_victim)), total=len(pm_victim)
                            ) if verbose else enumerate(pm_victim)
            # iterate through multiple behaviors
            for i in progress:
                self.workers[0](pm_victim[i], "logits", self.workers[0].model, ctrl_cand_ids)
                input_logit, input_id = self.workers[0].results.get()
                loss_target_ = pm_victim[i].target_loss(input_logit, input_id)
                loss_target += loss_target_.to(main_device)
                # loss_target[j*batch_size:(j+1)*batch_size] += (-loss_target_).exp()
                # loss_control_ = pm_victim[i].control_loss(last_tok_ll[0], input_id)
                loss_control_ = pm_victim[i].control_loss(last_tok_lls[i], input_id)
                loss_control += loss_control_.to(main_device)
                if verbose:
                    progress.set_description(f"loss_target={loss_target.min().item()/(i+1):.4f} | "
                                             f"loss_control={loss_control.min().item()/(i+1):.4f}")

        loss_target = loss_target / len(pm_victim)  # normalize by the # of behaviors
        # loss_target = loss_target.log_softmax(dim=1)  # re-normalize after aggregating over multiple behaviors
        # loss_target = - loss_target.log()  # geometric avg
        loss_control = loss_control / len(pm_victim)  # normalize by the # of behaviors
        # loss_control = loss_control.log_softmax(dim=1)  # re-normalize after aggregating over multiple behaviors

        # monitor the std of two losses to see if one dominates another during selection
        # print('target std', loss_target_j.std().item())
        # print('control std', loss_control_j.std().item())

        # optional heuristic: varying target weight for more diverse output
        if diversity_coef is None:
            diversity_coef = 1.
            # diversity_coef = 1 + 0.5 * torch.distributions.Beta(0.5, 2).sample().item()
        # soft-start the target objective
        # warm_up_w = min(len(pm_victim[0].control_toks) / 5, 1)
        warm_up_w = 1.

        loss = loss_control + w_tar_2 * diversity_coef * warm_up_w * loss_target  # * factor

        self.vocab_tracker.update(last_tok_cand_ids.squeeze(),
                                  loss_target,
                                  loss_control,
                                  loss,
                                  ctrl_cand_ids)

        top1_tok_id, _, _, _, _ = self.vocab_tracker.sample(1, 0)
        (sampled_last_tok_id,
         sampled_ctrl_id,
         sampled_target_loss,
         sampled_control_loss,
         sampled_loss) = self.vocab_tracker.sample(batch_size, ctrl_temp)
        next_ctrl = truthful_decode(tokenizer, sampled_ctrl_id)
        next_ctrl_last_tok = tokenizer.convert_ids_to_tokens(sampled_last_tok_id)

        print(f"Explored tokens:       {len(self.vocab_tracker.get_explored_ids())}")
        print(f"Current actual length:", len(tokenizer(next_ctrl).input_ids[1:]))
        print(f"Last token is         ", next_ctrl_last_tok)
        print(next_ctrl)

        # print candidate tokens for debugging
        topk_token_indices, topk_probabilities = self.vocab_tracker.get_top_k_tokens(10, ctrl_temp)
        topk_tokens = [tokenizer.convert_ids_to_tokens([topk_token_index])[0]
                       for topk_token_index in topk_token_indices]
        formatted_probs = [f"{prob:.4f}" for prob in topk_probabilities]
        max_width = max(max(len(token) for token in topk_tokens),
                        max(len(prob) for prob in formatted_probs))
        formatted_tokens = [f"{token:>{max_width}}" for token in topk_tokens]
        formatted_probs = [f"{prob:>{max_width}}" for prob in formatted_probs]
        print("Top-10 candidate tokens:")
        print(" ".join(formatted_tokens))
        print(" ".join(formatted_probs))

        # sanity check for decoding
        assert token_to_word(tokenizer, next_ctrl_last_tok) in ' ' + next_ctrl, \
            (f"Last control tok not in control! Last decoded control tok is "
             f"{token_to_word(tokenizer, next_ctrl_last_tok)}")

        return (next_ctrl,
                next_ctrl_last_tok,
                sampled_loss,
                sampled_target_loss,
                sampled_control_loss,
                top1_tok_id,
                diversity_coef)
