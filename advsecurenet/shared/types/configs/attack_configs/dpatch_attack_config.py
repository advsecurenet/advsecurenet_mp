from typing import Optional
from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs.attack_config import AttackConfig

@dataclass(kw_only=True)
class DPatchAttackConfig(AttackConfig):
    """
    DPatch attack configuration.
    """

    object_detector: str = "yolov5"
    patch_shape: tuple[int, int, int] = (3, 200, 200)
    learning_rate: float = 1.99
    max_iter: int = 800
    target_label: Optional[int] = None
