from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs.attack_config import AttackConfig


@dataclass(kw_only=True)
class DPatchAttackConfig(AttackConfig):
    """
    DPatch attack configuration.
    """

    object_detector: str = "fasterrcnn_resnet50_fpn"
    patch_shape: int = 10
    learning_rate: int = 10
    max_iter: int = 10
    batch_size: int = 10
    verbose: int = 10
    attack_feature: int = 10
