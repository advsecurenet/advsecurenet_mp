import os
import pkg_resources
from typing import Optional
import ssl
from contextlib import contextmanager

from advsecurenet.datasets.PascalVOC.pascalvoc_utils import voc_to_coco_anns
import torch
from torchvision import datasets

from advsecurenet.datasets.base_dataset import BaseDataset, DatasetWrapper
from advsecurenet.shared.normalization_params import NormalizationParameters
from advsecurenet.shared.types.configs.preprocess_config import PreprocessConfig
from advsecurenet.shared.types import DatasetType, DataType

@contextmanager
def temporarily_disable_ssl_verification():
    original_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        yield
    finally:
        ssl._create_default_https_context = original_context


class PascalVOCDataset(BaseDataset):
    """
    BaseDataset wrapper around torchvision.datasets.VOCDetection that
    RETURNS COCO-STYLE TARGETS via target_transform.
    """

    _SUPPORTED_YEARS = {"2007", "2012"}

    def __init__(self, preprocess_config: Optional[PreprocessConfig] = None):
        super().__init__(preprocess_config)

        # metadata
        self.name = "pascal_voc"
        self.num_classes = 20
        self.num_input_channels = 3

        # normalization (map to your ImageNet stats in NormalizationParameters)
        params = NormalizationParameters.get_params(DatasetType.PASCAL_VOC)
        self.mean = params.mean
        self.std = params.std

        # input size from preprocess config (fallback 224)
        if preprocess_config and getattr(preprocess_config, "steps", None):
            for step in preprocess_config.steps:
                if step.name.lower() == "resize" and "size" in step.params:
                    self.input_size = tuple(step.params["size"])
                    break
            else:
                self.input_size = (224, 224)
        else:
            self.input_size = (224, 224)

    def get_dataset_class(self):
        return datasets.VOCDetection

    @staticmethod
    def _validate_year(year: str):
        if str(year) not in PascalVOCDataset._SUPPORTED_YEARS:
            raise ValueError(
                f"Unsupported VOC year '{year}'. Supported: {PascalVOCDataset._SUPPORTED_YEARS}"
            )

    def load_dataset(
        self,
        root: Optional[str] = None,
        train: bool = True,
        download: bool = True,
        *,
        year: str = "2012",
        image_set: Optional[str] = None,
        **kwargs,
    ) -> DatasetWrapper:
        """
        Args:
            root: base dir (defaults to advsecurenet/data)
            train: True→'train', False→'val' unless image_set is provided
            download: let torchvision fetch/unpack VOC
            year: '2007' or '2012'
            image_set: 'train' | 'val' | 'trainval' | 'test' (overrides 'train')
        Returns:
            DatasetWrapper: wraps a VOCDetection that yields (image, COCO-style target)
        """
        # 1) root
        if root is None:
            root = pkg_resources.resource_filename("advsecurenet", "data")
        # 2) year & image_set
        self._validate_year(str(year))
        if image_set is None:
            image_set = "train" if train else "val"
        # 3) transforms
        transform = self.get_transforms()
        # 4) target_transform: VOC dict → COCO-style detection target
        def target_transform(target_dict):
            return voc_to_coco_anns(target_dict)
        # 5) instantiate
        with temporarily_disable_ssl_verification():
            voc_ds = datasets.VOCDetection(
                root=root,
                year=str(year),
                image_set=image_set,
                download=download,
                transform=transform,
                target_transform=target_transform,
                **kwargs,
            )
        # 6) wrap
        self._dataset = DatasetWrapper(dataset=voc_ds, name=self.name)
        # Treat train & trainval as TRAIN, everything else as TEST
        self.data_type = (
            DataType.TRAIN if image_set in {"train", "trainval"} else DataType.TEST
        )
        return self._dataset
