from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs import TOGAttackConfig
from cli.shared.types.attack import BaseAttackCLIConfigType, TargetedAttackCLIConfigType


@dataclass
class TOGAttackCLIConfigType(BaseAttackCLIConfigType):
    """
    This dataclass is used to store the configuration of the TOG attack CLI.
    """

    attack_config: TargetedAttackCLIConfigType[TOGAttackConfig]
