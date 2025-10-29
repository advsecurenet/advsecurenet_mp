from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs.attack_config import AttackConfig


@dataclass(kw_only=True)
class TOGAttackConfig(AttackConfig):
    """
    TOG attack configuration.
    """

    object_detector: str = "yolov5"
    max_iter: int = 130
    eps: float = 24 / 255.
    eps_iter: float = 2.0 / 255.0
    attack_type: str = "fabrication"
    mislabeling_mode: str = "ml"
