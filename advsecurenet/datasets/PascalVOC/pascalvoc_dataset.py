import os
import pkg_resources
from typing import Optional
import ssl
import logging
import tarfile
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
from contextlib import contextmanager

from advsecurenet.datasets.PascalVOC.pascalvoc_utils import voc_to_coco_anns
import torch
from torchvision import datasets

from advsecurenet.datasets.base_dataset import BaseDataset, DatasetWrapper
from advsecurenet.shared.normalization_params import NormalizationParameters
from advsecurenet.shared.types.configs.preprocess_config import PreprocessConfig
from advsecurenet.shared.types import DatasetType, DataType

logger = logging.getLogger(__name__)


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

    def _fallback_voc_download(self, root: str, year: str = "2012") -> None:
        os.makedirs(root, exist_ok=True)
        # If already extracted, skip downloading/extracting
        voc_dir = os.path.join(root, "VOCdevkit", f"VOC{year}")
        if os.path.isdir(voc_dir):
            logger.info(f"Found existing {voc_dir}, skipping mirror download.")
            return
        if str(year) == "2007":
            urls = [
                "https://data.brainchip.com/dataset-mirror/voc/VOCtrainval_06-Nov-2007.tar",
                "https://data.brainchip.com/dataset-mirror/voc/VOCtest_06-Nov-2007.tar",
            ]
        else:
            urls = [
                "https://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar",
            ]
        for url in urls:
            fname = os.path.join(root, os.path.basename(url))
            if not os.path.exists(fname):
                logger.info(f"Downloading {url} -> {fname}")
                try:
                    urlretrieve(url, fname)
                except (URLError, HTTPError) as e:
                    raise RuntimeError(f"Mirror download failed for {url}: {e}") from e
            # extract (plain .tar)
            logger.info(f"Extracting {fname} into {root}")
            with tarfile.open(fname, "r") as tf:
                tf.extractall(path=root)
        if not os.path.isdir(voc_dir):
            raise RuntimeError(
                f"Expected directory not found after extraction: {voc_dir}"
            )

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

        # # 4) target_transform: VOC dict → COCO-style detection target
        def target_transform(target_dict):
            anns = voc_to_coco_anns(target_dict)
            if len(anns) == 0:
                boxes = torch.empty((0, 4), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.int64)
            else:
                boxes = torch.as_tensor([a["bbox"] for a in anns], dtype=torch.float32)
                labels = torch.as_tensor([a["category_id"] for a in anns], dtype=torch.int64)
            return {"boxes": boxes, "labels": labels}

        # 5) instantiate
        with temporarily_disable_ssl_verification():
            try:
                voc_ds = datasets.VOCDetection(
                    root=root,
                    year=str(year),
                    image_set=image_set,
                    download=download,
                    transform=transform,
                    target_transform=target_transform,
                    **kwargs,
                )
            except Exception as e:
                if download:
                    logger.warning(
                        f"torchvision VOC download failed: {e}. Falling back to mirror..."
                    )
                    self._fallback_voc_download(root, year=str(year))
                    voc_ds = datasets.VOCDetection(
                        root=root,
                        year=str(year),
                        image_set=image_set,
                        download=False,
                        transform=transform,
                        target_transform=target_transform,
                        **kwargs,
                    )
                else:
                    raise
        # 6) wrap
        self._dataset = DatasetWrapper(dataset=voc_ds, name=self.name)
        # Treat train & trainval as TRAIN, everything else as TEST
        self.data_type = (
            DataType.TRAIN if image_set in {"train", "trainval"} else DataType.TEST
        )
        return self._dataset
