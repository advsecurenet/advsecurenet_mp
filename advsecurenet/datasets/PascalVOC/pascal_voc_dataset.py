from typing import Optional

from advsecurenet.datasets.base_dataset import BaseDataset, ImageFolderBaseDataset
from advsecurenet.shared.normalization_params import NormalizationParameters
from advsecurenet.shared.types.configs.preprocess_config import PreprocessConfig


class PascalVOCDataset(ImageFolderBaseDataset, BaseDataset):
    """s
    The PascalVOCDataset class that loads the Pascal VOC dataset.

    Attributes:
        
    """

    def __init__(self, preprocess_config: Optional[PreprocessConfig] = None):
        ImageFolderBaseDataset.__init__(self)
        BaseDataset.__init__(self, preprocess_config)
        self.mean = NormalizationParameters.get_params("PascalVOC").mean
        self.std = NormalizationParameters.get_params("PascalVOC").std
        self.input_size = (None, None) # TODO - update with correct values
        self.crop_size = (None, None) # TODO - update with correct values
        self.name = "pascal_voc"
        self.num_classes = None # TODO - update with correct values
        self.num_input_channels = None # TODO - update with correct values
