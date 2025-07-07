from typing import Tuple, Optional, Any
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

from advsecurenet.shared.types.configs.train_config import DifferentialPrivacyConfig

def setup_privacy_engine(
    model: nn.Module,
    optimizer: optim.Optimizer,
    data_loader: DataLoader,
    differential_privacy_config: DifferentialPrivacyConfig,
) -> Tuple[nn.Module, optim.Optimizer, DataLoader, PrivacyEngine, Optional[Any]]:
    """
    Initializes and attaches the Opacus PrivacyEngine to the training components.
    This function can handle both standard and "fast" clipping modes.

    Args:
        model (nn.Module): The model to be made private.
        optimizer (optim.Optimizer): The optimizer to be made private.
        data_loader (DataLoader): The data loader to be made private.
        differential_privacy_config (DifferentialPrivacyConfig): The configuration object with DP parameters.

    Returns:
        A tuple containing the wrapped (private) model, optimizer, data loader,
        the PrivacyEngine instance, and a new loss function if fast clipping is used (otherwise None).
    """
    privacy_engine = PrivacyEngine()
    
    # Correctly unpack kwargs for the make_private call
    kwargs = differential_privacy_config.kwargs or {}

    # The make_private method returns a variable number of items.
    # We capture them all in a tuple to inspect the result.
    results = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=differential_privacy_config.noise_multiplier,
        max_grad_norm=differential_privacy_config.max_grad_norm,
        **kwargs,
    )
    
    # Check if fast clipping was used by checking the number of returned items
    if kwargs.get("clipping") == "fast":
        if len(results) != 4:
            raise ValueError("Opacus with 'fast' clipping did not return the expected 4 values.")
        # Unpack the 4 values: model, optimizer, loss_fn, data_loader
        private_model, private_optimizer, private_loss_fn, private_data_loader = results
    else:
        if len(results) != 3:
            raise ValueError("Opacus did not return the expected 3 values.")
        # Unpack the 3 values and set loss_fn to None
        private_model, private_optimizer, private_data_loader = results
        private_loss_fn = None

    return private_model, private_optimizer, private_data_loader, privacy_engine, private_loss_fn