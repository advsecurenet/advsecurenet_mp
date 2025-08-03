import logging

from advsecurenet.shared.types.configs import ConfigType
from cli.logic.attack.attacker import CLIAttacker
from cli.logic.attack.od_attacker import CLIODAttacker
from cli.shared.types.attack import BaseAttackCLIConfigType
from cli.shared.utils.attack_mappings import attack_cli_mapping
from cli.shared.utils.config import load_and_instantiate_config

logger = logging.getLogger(__name__)

OD_ATTACKS = {"DPATCH", "TOG"}


def cli_attack(attack_name: str, config: str, **kwargs) -> None:
    """
    Entry point for the attack CLI. This function is called when the user wants to execute an attack. It determines the
    attack type and maps it to the respective configuration class. It then loads the configuration and executes the
    attack.

    Args:
        attack_name (str): The name of the attack to execute.
        config (str): The path to the configuration file.
        **kwargs: Additional keyword arguments to pass to the configuration class.

    Raises:
        ValueError: If the attack type is unknown.
    """
    if attack_name not in attack_cli_mapping:
        logger.error("Unknown attack type %s", attack_name)
        raise ValueError(f"Unknown attack type: {attack_name}")

    od_main_attack_type, attack_config_class = attack_cli_mapping[attack_name]

    config_data: BaseAttackCLIConfigType = load_and_instantiate_config(
        config=config,
        default_config_file=f"{attack_name.lower()}_attack_config.yml",
        config_type=ConfigType.ATTACK,
        config_class=attack_config_class,
        **kwargs,
    )
    logger.info("Loaded attack configuration: %s", config_data)
    try:
        if attack_name in OD_ATTACKS:
            attacker = CLIODAttacker(config_data, od_main_attack_type, **kwargs)
        else:
            attacker = CLIAttacker(config_data, od_main_attack_type, **kwargs)
        attacker.execute()
        logger.info("Attack completed successfully")
    except Exception as e:
        logger.error("Failed to execute attack: %s", e)
        raise e
