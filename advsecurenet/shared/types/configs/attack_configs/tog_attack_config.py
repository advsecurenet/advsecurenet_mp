from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs.attack_config import AttackConfig


@dataclass(kw_only=True)
class TOGAttackConfig(AttackConfig):
    """
    TOG attack configuration.
    """

    object_detector: str = "yolov5"
    max_iter: int = 60
    eps: float = 0.03
    eps_iter: float = 0.01
    attack_type: str = "mislabeling"
    mislabeling_mode: str = "ml"
