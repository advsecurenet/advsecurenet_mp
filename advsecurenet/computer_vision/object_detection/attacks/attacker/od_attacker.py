import logging
import abc
import numpy as np
import torch
from tqdm.auto import tqdm

from advsecurenet.evaluation.adversarial_evaluator import AdversarialEvaluator
from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.shared.types.configs.attack_configs.od_attacker_config import ODAttackerConfig

logger = logging.getLogger(__name__)

class ODAttacker(abc.ABC):
    """
    Orchestrates an object-detection attack where:
      - `config.attack.object_detector` is used to *generate* the patch
      - `config.model` is used to *detect* on the patched images
    """
    def __init__(self, config: ODAttackerConfig):
        self._config       = config
        self._device       = self._setup_device()
        self._eval_model   = config.model.to(self._device).eval()
        self._dataloader   = self._create_dataloader()


    def _setup_device(self):
        """
        Sets up the device for computation based on the configuration.
        Returns:
            torch.device: The device to be used for computation.
        """
        if self._config.device.processor:
            return torch.device(self._config.device.processor)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu") # replace with setup device utility function - TODO


    def _create_dataloader(self):
        """
        Creates a DataLoader instance based on the configuration.
        Returns:
            torch.utils.data.DataLoader: The DataLoader instance.
        """
        dl = self._config.dataloader
        if isinstance(dl, torch.utils.data.DataLoader):
            return dl
        return DataLoaderFactory.create_dataloader(dl)


    @abc.abstractmethod
    def execute(self):
        """
        Generate adversarial samples (or other attack outputs).
        Must be overridden in subclasses.
        """
        raise NotImplementedError("Subclasses of ODAttackerBase must implement execute().")
