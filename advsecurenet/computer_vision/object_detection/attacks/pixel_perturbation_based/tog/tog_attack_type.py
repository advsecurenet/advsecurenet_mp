from enum import Enum

class TOGAttackType(Enum):
    VANISHING = "vanishing"
    FABRICATION = "fabrication"
    MISLABELING = "mislabeling"
    UNTARGETED = "untargeted"