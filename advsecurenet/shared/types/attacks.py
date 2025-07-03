from enum import Enum

from advsecurenet.computer_vision.image_classification.attacks import FGSM, LOTS, PGD, CWAttack, DecisionBoundary, DeepFool
from advsecurenet.computer_vision.object_detection.attacks import DPatch, TOG


class AttackType(Enum):
    """
    This Enum class is used to store the types of attacks.
    """

    LOTS = LOTS
    FGSM = FGSM
    PGD = PGD
    CW = CWAttack
    DEEPFOOL = DeepFool
    DECISION_BOUNDARY = DecisionBoundary
    DPATCH = DPatch
    TOG = TOG
