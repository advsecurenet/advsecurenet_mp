import argparse
import yaml
import os
import sys

from advsecurenet.llm_new.core.attack_manager.attack_registry import ATTACKS
from advsecurenet.llm_new.core.attack_manager.attack_manager import get_workers, get_goals_and_targets
from advsecurenet.llm_new.core.prompts.attack_prompt import AttackPrompt
from advsecurenet.llm_new.core.prompts.prompt_manger import PromptManager

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_model_registry(registry_path):
    with open(registry_path, 'r') as f:
        return yaml.safe_load(f)

def setup_autodan_config(config, experiment_type):
    """Configure AutoDAN-specific settings based on experiment type"""
    
    # Set defaults for AutoDAN attacks
    autodan_defaults = {
        'n_steps': 1000,
        'batch_size': 512,
        'topk': 512,
        'progressive_goals': False,
        'stop_on_success': True,
        'allow_non_ascii': True,
        'n_test_data': 25,
        'test_offset': 25,
        'w_tar_1': 0.4,
        'w_tar_2': 100,
        'ctrl_temp': 1,
        'insert_loc': 'tail',
        'ctrl_prefix': 'vicuna',
        'ctrl_suffix': 'empty',
        'budget': 1,
        'n_sample_goals': 0,
        'dynamic_weight': False
    }
    
    # Experiment-specific configurations
    experiment_configs = {
        'individual': {
            'attack_name': 'gcg',
            'config_prefix': 'individual',
            'train_data': '../../data/advbench/harmful_behaviors.csv',
            'num_train_models': 1,
            'budget': 3,
            'test_steps': 50,
            'batch_size': 512
        },
        'transfer': {
            'attack_name': 'autodan',
            'config_prefix': 'transfer',
            'train_data': '../../data/advbench/harmful_behaviors.csv',
            'num_train_models': 1,
            'budget': 1,
            'test_steps': 25,
            'batch_size': 512
        },
        'mixture': {
            'attack_name': 'autodan',
            'config_prefix': 'mixture',
            'train_data': '../../data/advbench/harmful_behaviors.csv',
            'num_train_models': 2,
            'budget': 1,
            'test_steps': 25,
            'batch_size': 512
        },
        'leak': {
            'attack_name': 'autodan',
            'config_prefix': 'transfer',
            'train_data': '../../data/prompt_leaking/aws_prompts.csv',
            'num_train_models': 1,
            'budget': 1,
            'test_steps': 10,
            'batch_size': 512
        },
        'evaluate': {
            'attack_name': 'autodan',
            'config_prefix': 'transfer',
            'train_data': '../../data/advbench/harmful_behaviors.csv',
            'num_train_models': 1,
            'budget': 1,
            'test_steps': 25,
            'batch_size': 64,
            'n_train_data': 25
        }
    }
    
    # Apply defaults
    for key, value in autodan_defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    
    # Apply experiment-specific config
    if experiment_type in experiment_configs:
        exp_config = experiment_configs[experiment_type]
        for key, value in exp_config.items():
            setattr(config, key, value)
    
    # Generate result prefix
    if not hasattr(config, 'result_prefix'):
        config.result_prefix = f"../results/{experiment_type}_{config.model['name']}_{config.n_train_data}_{config.insert_loc}_{config.w_tar_1}_{config.w_tar_2}_{config.ctrl_temp}"
    
    return config

def run_attack(config):
    # === Load model registry and set model/tokenizer/template ===
    model_registry_path = os.path.join(os.path.dirname(__file__), '..', 'model_registry.yaml')
    model_registry = load_model_registry(model_registry_path)
    model_name = config.model['name']
    model_info = model_registry[model_name]
    config.model_paths = [os.path.expandvars(model_info['path'])]
    config.tokenizer_paths = [os.path.expandvars(model_info['tokenizer'])]
    config.conversation_templates = [model_info['template']]
    config.tokenizer_kwargs = [{}] 
    config.model_kwargs = [{}]     
    config.devices = [config.model.get('device', 'cpu')] 

    # === Load models ===
    train_workers, test_workers = get_workers(config)

    # === Load goals and targets ===
    goals, targets, test_goals, test_targets = get_goals_and_targets(config)

    # === Setup managers ===
    managers = {
        "AP": AttackPrompt,
        "PM": PromptManager,
    }

    # === Initialize attack ===
    attack_class = ATTACKS[config.attack_name]
    attack = attack_class(
        goals,
        targets,
        train_workers,
        control_init=config.control_init,
        test_prefixes=config.test_prefixes,
        logfile=config.logfile,
        managers=managers,
        test_goals=test_goals,
        test_targets=test_targets,
        test_workers=test_workers
    )

    # === Run attack ===
    attack.run(
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        topk=config.topk,
        temp=getattr(config, 'temp', 1.0),
        target_weight=getattr(config, 'target_weight', 1.0),
        control_weight=getattr(config, 'control_weight', 1.0),
        stop_on_success=config.stop_on_success,
        test_steps=config.test_steps,
        filter_cand=getattr(config, 'filter_cand', True),
        verbose=getattr(config, 'verbose', True)
    )

    # === Cleanup ===
    for worker in train_workers + test_workers:
        worker.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified AutoDAN Attack Script')
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    parser.add_argument("--experiment_type", type=str, 
                       choices=['individual', 'transfer', 'mixture', 'leak', 'evaluate'],
                       default='transfer', help="Type of experiment to run")
    parser.add_argument("--data_offset", type=int, default=0, 
                       help="Data offset for individual experiments")
    args = parser.parse_args()

    # Set environment variables
    os.environ['WANDB_MODE'] = 'disabled'

    # Load and configure
    config_dict = load_config(args.config)
    
    # Convert config dict to namespace for dot-access
    class ConfigNamespace:
        def __init__(self, d): 
            self.__dict__.update(d)
    
    config = ConfigNamespace(config_dict)
    
    # Setup AutoDAN configuration
    config = setup_autodan_config(config, args.experiment_type)
    
    # Add data offset if specified
    if hasattr(config, 'data_offset') or args.data_offset > 0:
        config.data_offset = args.data_offset
    
    # Create results directory
    results_dir = "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    
    # Run the attack
    if args.experiment_type == 'individual':
        # Run individual experiment with data offset loop
        for data_offset in [0]:  # Add more offsets if needed: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
            config.data_offset = data_offset
            run_attack(config)
    else:
        run_attack(config)