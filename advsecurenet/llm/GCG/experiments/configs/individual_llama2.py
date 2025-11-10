import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_llama2'

    # Use GPT-2 for testing (publicly available)
    config.tokenizer_paths=["gpt2"]
    config.model_paths=["gpt2"]
    config.conversation_templates=['zero_shot']  # GPT-2 doesn't use llama-2 template
    config.devices=["cpu"]

    config.model_kwargs=[{
        "low_cpu_mem_usage": True, 
        "use_cache": False
    }]

    return config