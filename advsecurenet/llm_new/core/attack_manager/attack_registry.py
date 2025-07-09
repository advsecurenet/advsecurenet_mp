# advsecurenet/llm/core/attack_manager/attack_registry.py

from advsecurenet.llm_new.attacks.gcg.attack import GCGMultiPromptAttack

ATTACKS = {
    "gcg": GCGMultiPromptAttack
}