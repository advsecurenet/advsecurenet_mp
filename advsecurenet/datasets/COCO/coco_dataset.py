import os
import pkg_resources
from typing import Optional

from torchvision import datasets
from torchvision.datasets.utils import download_and_extract_archive

from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.datasets.base_dataset import DatasetWrapper
from advsecurenet.shared.normalization_params import NormalizationParameters
from advsecurenet.shared.types.configs.preprocess_config import PreprocessConfig
from advsecurenet.shared.types import DatasetType, DataType


class COCODataset(BaseDataset):
    """
    A BaseDataset wrapper around torchvision.datasets.CocoDetection
    """

    def __init__(self, preprocess_config: Optional[PreprocessConfig] = None):
        super().__init__(preprocess_config)

        # metadata
        self.name = "coco"
        self.num_classes = 80
        self.num_input_channels = 3

        # normalization (we map COCO→ImageNet stats in your NormalizationParameters)
        params = NormalizationParameters.get_params(DatasetType.COCO)
        self.mean = params.mean
        self.std = params.std

    @staticmethod
    def _maybe_download_coco(root: str, train: bool):
        split = "train2017" if train else "val2017"
        # image archive & annotation archive
        img_url = f"http://images.cocodataset.org/zips/{split}.zip"
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

        img_dir = os.path.join(root, split)
        ann_dir = os.path.join(root, "annotations")

        if not os.path.isdir(img_dir):
            download_and_extract_archive(url=img_url, download_root=root, extract_root=root)

        if not os.path.isdir(ann_dir):
            download_and_extract_archive(url=ann_url, download_root=root, extract_root=root)

    def get_dataset_class(self):
        return datasets.CocoDetection

    def load_dataset(
        self,
        root: Optional[str] = None,
        train: bool = True,
        download: bool = True,
        **kwargs,
    ) -> DatasetWrapper:
        """
        Overrides BaseDataset.load_dataset to handle COCO download + instantiation.

        Args:
            root (str, optional): base dir for COCO. Defaults to advsecurenet/data.
            train (bool, optional): use train2017 vs val2017. Defaults to True.
            download (bool, optional): whether to fetch/unzip COCO. Defaults to True.
            **kwargs: unused for COCO.

        Returns:
            DatasetWrapper: wraps a torchvision.datasets.CocoDetection
        """
        # 1) set up root
        if root is None:
            root = pkg_resources.resource_filename("advsecurenet", "data")

        # 2) download if requested
        if download:
            self._maybe_download_coco(root, train)

        # 3) build image folder + annotation path
        split = "train2017" if train else "val2017"
        img_root = os.path.join(root, split)
        ann_file = os.path.join(root, "annotations", f"instances_{'train' if train else 'val'}2017.json")

        # 4) get transforms from preprocess_config
        transform = self.get_transforms()

        # 5) instantiate the torchvision dataset
        coco_ds = datasets.CocoDetection(root=img_root, annFile=ann_file, transform=transform)

        # 6) wrap and return
        self._dataset = DatasetWrapper(dataset=coco_ds, name=self.name)
        self.data_type = DataType.TRAIN if train else DataType.TEST
        return self._dataset
