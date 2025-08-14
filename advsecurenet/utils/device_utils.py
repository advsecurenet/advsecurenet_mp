import torch


@staticmethod
def move_batch_to_device(images, targets_dict, device):
    def move_to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, dict):
            return {k: move_to_device(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [move_to_device(v) for v in x]
        else:
            return x

    return move_to_device(images), move_to_device(targets_dict)


@staticmethod
def setup_device(processor):
    """
    Sets up the device for computation based on the configuration.
    Returns:
        torch.device: The device to be used for computation.
    """
    if processor:
        return torch.device(processor)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
