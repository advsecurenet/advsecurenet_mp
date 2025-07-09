import torch
import torch.nn as nn

from advsecurenet.llm_new.core.models.embeddings import get_embedding_matrix, get_embeddings


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):
    """
    Computes gradients of the loss w.r.t. selected tokens in the input.

    Parameters
    ----------
    model : torch.nn.Module
        The language model used for generation.
    input_ids : torch.Tensor
        Tokenized input IDs (1D tensor).
    input_slice : slice
        Slice of `input_ids` over which to apply adversarial control.
    target_slice : slice
        Slice used as labels for computing loss.
    loss_slice : slice
        Slice used to extract relevant logits for the loss.

    Returns
    -------
    torch.Tensor
        Gradients of shape [input_slice_length, vocab_size]
    """

    embed_weights = get_embedding_matrix(model)

    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()

    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # Stitch with frozen (non-adversarial) embeddings
    full_embeds = torch.cat([
        get_embeddings(model, input_ids.unsqueeze(0)).detach()[:, :input_slice.start, :],
        input_embeds,
        get_embeddings(model, input_ids.unsqueeze(0)).detach()[:, input_slice.stop:, :]
    ], dim=1)

    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)

    loss.backward()
    if one_hot.grad is None:
        raise RuntimeError("Gradient was not computed. Make sure one_hot is a leaf tensor with requires_grad=True and loss.backward() was called.")
    return one_hot.grad.clone()
