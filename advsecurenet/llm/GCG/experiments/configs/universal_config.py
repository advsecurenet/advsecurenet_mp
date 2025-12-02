import os
import torch
from ml_collections import config_dict

def get_config():
    """
    Universal GCG config that works with any HuggingFace model.
    Uses the ConversationTemplateAdapter in attack_manager.py for template handling.
    All parameters can be overridden via command line using --config.parameter_name=value
    """
    
    # Start with fresh config
    config = config_dict.ConfigDict()
    
    # === UNIVERSAL MODE SETTINGS ===
    config.universal_mode = True        # Enable universal model support
    config.auto_template_detection = False  # Let attack_manager handle templates
    
    # === MODEL PARAMETERS ===
    config.model_name = "microsoft/DialoGPT-small"  # Default - can be overridden
    config.device = "auto"
    
    # === ATTACK PARAMETERS ===  
    config.attack_type = "individual"  # "individual" or "transfer"
    config.data_type = "behaviors"     # "behaviors" or "strings"
    config.attack = "gcg"              # Attack method
    config.transfer = False
    
    # === DATA PARAMETERS ===
    config.n_train_data = 10
    config.n_test_data = 0
    config.data_offset = 0
    config.train_data = '/Users/philip/Desktop/advsecurenet_mp/advsecurenet/llm/GCG/data/advbench/harmful_behaviors.csv'
    config.test_data = ''
    
    # === OPTIMIZATION PARAMETERS ===
    config.n_steps = 100
    config.test_steps = 10
    config.batch_size = 256
    config.lr = 0.1
    config.topk = 256
    config.temp = 1.5
    
    # === CONTROL PARAMETERS ===
    config.control_init = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
    config.control_weight = 0.0
    config.target_weight = 1.0
    
    # === PROGRESSIVE PARAMETERS ===
    config.progressive_goals = False
    config.progressive_models = False
    config.incr_control = False
    
    # === OTHER PARAMETERS ===
    config.anneal = False
    config.stop_on_success = False
    config.filter_cand = True
    config.gbda_deterministic = True
    config.allow_non_ascii = False
    config.verbose = True
    config.num_train_models = 1
    
    # === ADVANCED MODEL PARAMETERS ===
    config.low_cpu_mem_usage = True
    config.use_cache = False
    config.use_fast_tokenizer = False
    
    # === COMPUTED FIELDS ===
    # Auto-detect device if needed
    if config.device == "auto":
        if torch.cuda.is_available():
            config.device = "cuda:0"
        else:
            config.device = "cpu"
    
    # === REQUIRED DERIVED FIELDS (set immediately) ===
    # These MUST exist for the framework to work
    def get_current_model_name():
        return getattr(config, 'model_name', 'gpt2')
    config.model_paths = (config.model_name,)  # Tuple instead of list
    config.tokenizer_paths = (config.model_name,)  # Tuple instead of list  
    config.devices = (config.device,)  # Tuple instead of list
    config.conversation_templates = ("zero_shot",)  # Tuple instead of list
    
    # === MODEL KWARGS ===
    config.model_kwargs = [{"low_cpu_mem_usage": config.low_cpu_mem_usage, "use_cache": config.use_cache}]
    config.tokenizer_kwargs = [{"use_fast": config.use_fast_tokenizer}]

    # === RESULTS PATH ===
    config.result_prefix = f'../results/{config.attack_type}_{config.data_type}_{_clean_model_name(get_current_model_name())}_gcg_offset{config.data_offset}'
    
    # Ensure results directory exists
    results_dir = os.path.dirname(config.result_prefix) 
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
        print(f"🔧 Created results directory: {results_dir}")
    
    # === DEBUG INFO ===
    if config.verbose:
        print(f"🔧 Universal GCG Config Loaded")
        print(f"🔧 Model: {config.model_name}")
        print(f"🔧 Model Paths: {config.model_paths}")
        print(f"🔧 Tokenizer Paths: {config.tokenizer_paths}")
        print(f"🔧 Device: {config.device}")
        print(f"🔧 Universal Mode: {config.universal_mode}")
        print(f"🔧 Template will be auto-adapted by ConversationTemplateAdapter")
    
    return config

def _clean_model_name(model_name):
    """Clean model name for use in file paths."""
    return model_name.replace("/", "_").replace("-", "_")