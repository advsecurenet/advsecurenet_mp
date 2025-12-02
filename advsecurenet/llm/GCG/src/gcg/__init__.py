"""
GCG (Greedy Coordinate Gradient) attack implementation.
Exposes the sophisticated GCG classes for dynamic import.
"""

# Import only the classes that actually exist in gcg_attack.py
from .gcg_attack import (
    GCGAttackPrompt,
    GCGPromptManager,
    GCGMultiPromptAttack,
    token_gradients,
)

# Import fallbacks for classes that don't exist in GCG
from ..attacks.individual import IndividualPromptAttack

# Alias them to the expected names for dynamic import
AttackPrompt = GCGAttackPrompt
PromptManager = GCGPromptManager
MultiPromptAttack = GCGMultiPromptAttack

# Use the regular individual attack since GCG doesn't have its own
# Use MultiPromptAttack for progressive since GCG doesn't have a separate one
ProgressiveMultiPromptAttack = GCGMultiPromptAttack

# Export all classes
__all__ = [
    "AttackPrompt",
    "PromptManager",
    "MultiPromptAttack",
    "IndividualPromptAttack",
    "ProgressiveMultiPromptAttack",
    "GCGAttackPrompt",
    "GCGPromptManager",
    "GCGMultiPromptAttack",
    "token_gradients",
]
