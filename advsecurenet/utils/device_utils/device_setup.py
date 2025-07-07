from typing import Optional
import torch

def setup_device(processor: Optional[str]) -> torch.device:
    """
    Sets up the device for training by auto-detecting or using the specified processor.

    Args:
        processor (Optional[str]): The device to use, e.g., "cuda", "cpu", "mps". 
                                   If None, it will auto-detect available hardware.

    Returns:
        torch.device: The selected torch.device object.
    """
    if processor:
        device = torch.device(processor)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    return device