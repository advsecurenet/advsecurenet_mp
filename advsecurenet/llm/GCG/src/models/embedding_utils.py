from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel,
                          GPTJForCausalLM, GPTNeoXForCausalLM,
                          LlamaForCausalLM)
import torch, json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_embedding_layer(model):
    if isinstance(model, GPTJForCausalLM) or isinstance(model, GPT2LMHeadModel):
        return model.transformer.wte
    elif isinstance(model, LlamaForCausalLM):
        return model.model.embed_tokens
    elif isinstance(model, GPTNeoXForCausalLM):
        return model.base_model.embed_in
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        return model.transformer.wte  # GPT-2, GPT-J, etc.
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens  # LLaMA, Mistral, etc.
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'embed_in'):
        return model.gpt_neox.embed_in  # GPT-NeoX variants
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'word_embeddings'):
        return model.transformer.word_embeddings  # BERT-style
    elif hasattr(model, 'embeddings'):
        return model.embeddings.word_embeddings  # Some other architectures
    else:
        # Fallback: search for embedding layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding) and 'embed' in name.lower():
                return module
        raise ValueError(f"Could not find embedding layer for model type: {type(model)}")
        

def get_embedding_matrix(model):
    return get_embedding_layer(model).weight

def get_embeddings(model, input_ids):
    embedding_layer = get_embedding_layer(model)
    if hasattr(embedding_layer.weight, 'half'):
        return embedding_layer(input_ids).half()
    return embedding_layer(input_ids)

def get_nonascii_toks(tokenizer, device='cpu'):

    def is_ascii(s):
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
