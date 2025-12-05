"""Embedding utilities for language model manipulation and analysis.

This module provides utilities for extracting and manipulating embedding layers
from various transformer architectures. It includes functions for:
- Finding embedding layers across different model architectures
- Extracting embedding matrices and computing embeddings
- Identifying non-ASCII tokens for adversarial attacks
- JSON encoding utilities for numpy objects

Supported architectures include GPT-2, GPT-J, GPT-NeoX, LLaMA, Mistral,
and other transformer-based models with automatic fallback detection.
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPTJForCausalLM,
    GPTNeoXForCausalLM,
    LlamaForCausalLM,
)
import torch
import json
import numpy as np


class NpEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy data types.

    Extends the default JSON encoder to properly serialize numpy integers,
    floats, booleans, and arrays to their Python equivalents.

    This is particularly useful when saving experimental results or
    configurations that contain numpy objects.
    """

    def default(self, obj):
        """Convert numpy objects to JSON-serializable Python types.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable representation of the object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_embedding_layer(model):
    """Extract the embedding layer from various transformer model architectures.

    This function provides a unified interface to access embedding layers across
    different transformer architectures. It handles specific model types first,
    then falls back to generic attribute-based detection.

    Supported architectures:
    - GPT-2, GPT-J: model.transformer.wte
    - LLaMA, Mistral: model.model.embed_tokens
    - GPT-NeoX: model.base_model.embed_in or model.gpt_neox.embed_in
    - BERT-style: model.transformer.word_embeddings
    - Generic: searches for torch.nn.Embedding modules

    Args:
        model: HuggingFace transformer model instance

    Returns:
        torch.nn.Embedding: The embedding layer of the model

    Raises:
        ValueError: If no embedding layer can be found
    """
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in
    elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model.transformer.wte  # GPT-2, GPT-J, etc.
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens  # LLaMA, Mistral, etc.
    elif hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "embed_in"):
        return model.gpt_neox.embed_in  # GPT-NeoX variants
    elif hasattr(model, "transformer") and hasattr(
        model.transformer, "word_embeddings"
    ):
        return model.transformer.word_embeddings  # BERT-style
    elif hasattr(model, "embeddings"):
        return model.embeddings.word_embeddings  # Some other architectures
    else:
        # Fallback: search for embedding layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding) and "embed" in name.lower():
                return module
        raise ValueError(
            f"Could not find embedding layer for model type: {type(model)}"
        )


def get_embedding_matrix(model):
    """Extract the embedding weight matrix from a transformer model.

    Args:
        model: HuggingFace transformer model instance

    Returns:
        torch.Tensor: The embedding weight matrix with shape [vocab_size, hidden_size]
    """
    return get_embedding_layer(model).weight


def get_embeddings(model, input_ids):
    """Compute embeddings for input token IDs using the model's embedding layer.

    This function handles precision conversion automatically, using half precision
    when available for memory efficiency.

    Args:
        model: HuggingFace transformer model instance
        input_ids: Token IDs tensor with shape [batch_size, sequence_length]

    Returns:
        torch.Tensor: Embeddings tensor with shape [batch_size, sequence_length, hidden_size]
    """
    embedding_layer = get_embedding_layer(model)
    if hasattr(embedding_layer.weight, "half"):
        return embedding_layer(input_ids).half()
    return embedding_layer(input_ids)


def get_nonascii_toks(tokenizer, device="cpu"):
    """Identify non-ASCII and special tokens in the tokenizer vocabulary.

    This function is particularly useful for adversarial attacks where
    non-ASCII tokens might cause issues or need to be filtered out.
    It identifies tokens that are either non-ASCII or non-printable,
    as well as special tokens like BOS, EOS, PAD, and UNK.

    Args:
        tokenizer: HuggingFace tokenizer instance
        device: Device to place the resulting tensor on (default: "cpu")

    Returns:
        torch.Tensor: Tensor containing token IDs of non-ASCII and special tokens

    Note:
        Starts checking from token ID 3 to skip common special tokens at the beginning
        of the vocabulary.
    """

    def is_ascii(s):
        """Check if string contains only ASCII printable characters.

        Args:
            s: String to check

        Returns:
            bool: True if string is ASCII and printable
        """
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    return torch.tensor(ascii_toks, device=device)
