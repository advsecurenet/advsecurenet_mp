from dataclasses import dataclass

from advsecurenet.shared.types.configs.attack_configs.attack_config import AttackConfig


@dataclass(kw_only=True)
class TOGAttackConfig(AttackConfig):
    """
    TOG attack configuration.
    """

    object_detector: str = "yolov5"
    max_iter: int = 10
    eps: float = 0.1
    eps_iter: float = 0.1
    batch_size: int = 10
    verbose: int = 10
    attack_type: str = "vanishing"
    mislabeling_mode: str = "ml"
