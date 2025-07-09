# import argparse
# import yaml
# import os

# from advsecurenet.llm_new.core.attack_manager.attack_registry import ATTACKS
# from advsecurenet.llm_new.core.attack_manager.attack_manager import get_workers, get_goals_and_targets
# from advsecurenet.llm_new.core.prompts.attack_prompt import AttackPrompt
# from advsecurenet.llm_new.core.prompts.prompt_manger import PromptManager



# def load_config(config_path):
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)


# def run_attack(config):
#     # === Load models ===
#     train_workers, test_workers = get_workers(config)

#     # === Load goals and targets ===
#     goals, targets, test_goals, test_targets = get_goals_and_targets(config)

#     # === Setup managers ===
#     managers = {
#         "AP": AttackPrompt,
#         "PM": PromptManager,
#     }

#     # === Initialize attack ===
#     attack_class = ATTACKS[config.attack_name]
#     attack = attack_class(
#         goals,
#         targets,
#         train_workers,
#         control_init=config.control_init,
#         test_prefixes=config.test_prefixes,
#         logfile=config.logfile,
#         managers=managers,
#         test_goals=test_goals,
#         test_targets=test_targets,
#         test_workers=test_workers
#     )

#     # === Run attack ===
#     attack.run(
#         n_steps=config.n_steps,
#         batch_size=config.batch_size,
#         topk=config.topk,
#         temp=config.temp,
#         target_weight=config.target_weight,
#         control_weight=config.control_weight,
#         stop_on_success=config.stop_on_success,
#         #log_first=config.log_first,
#         test_steps=config.test_steps,
#         filter_cand=config.filter_cand,
#         verbose=config.verbose
#     )


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
#     args = parser.parse_args()

#     # Load and run
#     config_dict = load_config(args.config)

#     # Convert config dict to namespace for dot-access
#     class ConfigNamespace:
#         def __init__(self, d): self.__dict__.update(d)

#     config = ConfigNamespace(config_dict)
#     run_attack(config)


import argparse
import yaml
import os

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

def run_attack(config):
    # === Load model registry and set model/tokenizer/template ===
    model_registry_path = os.path.join(os.path.dirname(__file__), '..', 'model_registry.yaml')
    model_registry = load_model_registry(model_registry_path)
    model_name = config.model['name']  # e.g., 'llama2'
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
        temp=config.temp,
        target_weight=config.target_weight,
        control_weight=config.control_weight,
        stop_on_success=config.stop_on_success,
        test_steps=config.test_steps,
        filter_cand=config.filter_cand,
        verbose=config.verbose
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    # Load and run
    config_dict = load_config(args.config)

    # Convert config dict to namespace for dot-access
    class ConfigNamespace:
        def __init__(self, d): self.__dict__.update(d)

    config = ConfigNamespace(config_dict)
    run_attack(config)