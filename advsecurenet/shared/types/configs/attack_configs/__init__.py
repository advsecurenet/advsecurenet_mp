from advsecurenet.shared.types.configs.attack_configs.cw_attack_config import (
    CWAttackConfig,
)
from advsecurenet.shared.types.configs.attack_configs.decision_boundary_attack_config import (
    DecisionBoundaryAttackConfig,
)
from advsecurenet.shared.types.configs.attack_configs.deepfool_attack_config import (
    DeepFoolAttackConfig,
)
from advsecurenet.shared.types.configs.attack_configs.fgsm_attack_config import (
    FgsmAttackConfig,
)
from advsecurenet.shared.types.configs.attack_configs.lots_attack_config import (
    LotsAttackConfig,
    LotsAttackMode,
)
from advsecurenet.shared.types.configs.attack_configs.pgd_attack_config import (
    PgdAttackConfig,
)
from advsecurenet.shared.types.configs.attack_configs.dpatch_attack_config import (
    DPatchAttackConfig,
)
from advsecurenet.shared.types.configs.attack_configs.tog_attack_config import (
    TOGAttackConfig,
)

__all__ = [
    # image classification
    "CWAttackConfig",
    "DeepFoolAttackConfig",
    "FgsmAttackConfig",
    "LotsAttackConfig",
    "LotsAttackMode",
    "PgdAttackConfig",
    "DecisionBoundaryAttackConfig",
    # object detection
    "DPatchAttackConfig",
    "TOGAttackConfig",
]

__iter__ = __all__.__iter__
