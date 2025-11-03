import os
import torch
from ml_collections import config_dict

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    """
    Universal GCG config that works with any HuggingFace model.
    All parameters can be overridden via command line using --config.parameter_name=value
    """
    
    # Start with default config
    config = default_config()
    
    # === ADD ALL CONFIGURABLE FIELDS TO THE CONFIG ===
    # Model parameters
    config.model_name = "gpt2"
    config.device = "auto"
    
    # Attack parameters  
    config.attack_type = "individual"  # "individual" or "transfer"
    config.data_type = "behaviors"     # "behaviors" or "strings"
    config.attack = "gcg"              # Attack method
    
    # Data parameters (override defaults if they exist)
    config.n_train_data = 10
    config.n_test_data = 0
    config.data_offset = 0
    
    # Optimization parameters (override defaults if they exist)
    config.n_steps = 1000
    config.test_steps = 50
    config.batch_size = 512
    config.lr = 0.01
    config.topk = 256
    config.temp = 1
    
    # Control parameters (override defaults if they exist)
    config.control_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    config.control_weight = 0.0
    config.target_weight = 1.0
    
    # Progressive parameters (override defaults if they exist)
    config.progressive_goals = False
    config.progressive_models = False
    config.incr_control = False
    
    # Other parameters (override defaults if they exist)
    config.anneal = False
    config.stop_on_success = False
    config.filter_cand = True
    config.gbda_deterministic = True
    config.allow_non_ascii = False
    config.verbose = True
    config.num_train_models = 1
    
    # Advanced model parameters
    config.low_cpu_mem_usage = True
    config.use_cache = False
    config.use_fast_tokenizer = False
    
    # === COMPUTED FIELDS (derived from the above) ===
    # Auto-detect device if needed
    if config.device == "auto":
        if torch.cuda.is_available():
            config.device = "cuda:0"
        else:
            config.device = "cpu"
    
    # Auto-detect appropriate conversation template
    conversation_template = _get_conversation_template(config.model_name)
    
    # Set derived fields
    config.model_paths = [config.model_name]
    config.tokenizer_paths = [config.model_name]
    config.devices = [config.device]
    config.conversation_templates = [conversation_template]
    
    # Set attack-specific config
    config.transfer = (config.attack_type == "transfer")
    config.train_data = f'../../data/advbench/harmful_{config.data_type}.csv'
    config.test_data = ''
    
    # Model kwargs
    config.model_kwargs = _get_model_kwargs(config.device, config.low_cpu_mem_usage, config.use_cache)
    config.tokenizer_kwargs = [{"use_fast": config.use_fast_tokenizer}]
    
    # Results path
    config.result_prefix = f'../results/{config.attack_type}_{config.data_type}_{_clean_model_name(config.model_name)}_gcg_offset{config.data_offset}'
    
    return config

def _get_conversation_template(model_name):
    """Auto-detect appropriate conversation template based on model name."""
    model_lower = model_name.lower()
    
    if "llama-2" in model_lower or "llama2" in model_lower:
        return "llama-2"
    elif "vicuna" in model_lower:
        return "vicuna_v1.1"
    elif "mistral" in model_lower:
        return "mistral"
    elif "gpt" in model_lower:
        return "zero_shot"
    elif "claude" in model_lower:
        return "claude"
    else:
        return "zero_shot"

def _clean_model_name(model_name):
    """Clean model name for use in file paths."""
    return model_name.replace("/", "_").replace("-", "_")

def _get_model_kwargs(device, low_cpu_mem_usage=True, use_cache=False):
    """Get appropriate model kwargs based on device."""
    kwargs = {
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "use_cache": use_cache
    }
    return [kwargs]