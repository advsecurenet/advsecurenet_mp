import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_tiny_llama2'
    config.attack = 'gcg'
    config.transfer = False
    
    # Use DialoGPT-small which is working
    config.tokenizer_paths = ["microsoft/DialoGPT-small"]
    config.model_paths = ["microsoft/DialoGPT-small"]
    config.conversation_templates = ['llama-2']
    config.devices = ["cpu"]
    
    # Data configuration
    config.train_data = '../data/advbench/harmful_behaviors.csv'
    config.test_data = '../data/advbench/harmful_behaviors.csv'
    
    # Adjusted parameters to avoid edge cases
    config.n_train_data = 1
    config.n_test_data = 0
    config.data_offset = 0
    config.n_steps = 5        # Increased slightly
    config.test_steps = 2     # Test every 2 steps
    config.batch_size = 8     # Increased batch size
    config.topk = 64          # Reduced topk to avoid empty candidate lists
    
    # Control parameters
    config.control_init = "! ! ! ! ! ! ! !"  # Longer control string
    config.temp = 1
    config.target_weight = 1.0
    config.control_weight = 0.0
    config.lr = 0.01
    config.gbda_deterministic = True
    config.anneal = False
    config.incr_control = False
    config.stop_on_success = False
    config.verbose = True
    config.filter_cand = False  # Disable candidate filtering to avoid the error
    config.allow_non_ascii = False
    
    return config