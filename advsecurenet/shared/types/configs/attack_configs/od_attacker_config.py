from dataclasses import dataclass, field
from typing import Optional

from advsecurenet.shared.types.configs.attack_configs.attacker_config import AttackerConfig

@dataclass
class ODAttackerConfig(AttackerConfig):
    """
    Configuration class for the ODAttacker module.
    """

    target_label: Optional[int] = None
    evaluators: list[str] = field(default_factory=lambda: ["mean_average_precision"])
