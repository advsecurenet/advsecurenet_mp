from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs import DPatchAttackConfig
from cli.shared.types.attack import BaseAttackCLIConfigType, TargetedAttackCLIConfigType


@dataclass
class DPatchAttackCLIConfigType(BaseAttackCLIConfigType):
    """
    This dataclass is used to store the configuration of the DPatch attack CLI.
    """

    attack_config: TargetedAttackCLIConfigType[DPatchAttackConfig]
