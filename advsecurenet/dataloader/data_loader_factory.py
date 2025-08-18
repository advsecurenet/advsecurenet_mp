"""
This module contains the DataLoaderFactory class that creates a DataLoader for the given dataset.
"""

import torch

from dataclasses import asdict, is_dataclass
from typing import Optional, Callable

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.distributed import DistributedSampler

from advsecurenet.shared.types.configs.dataloader_config import DataLoaderConfig
from advsecurenet.datasets.COCO.coco_utils import map_raw_to_contiguous


def dataclass_to_dict(instance):
    """
    Convert a dataclass instance to a dictionary, handling non-serializable fields appropriately.
    """
    result = {}
    for field in instance.__dataclass_fields__:
        value = getattr(instance, field)
        # Handle special cases for non-serializable fields if necessary
        if isinstance(value, (DistributedSampler, TorchDataset)):
            result[field] = value
        elif is_dataclass(value):
            result[field] = asdict(value)
        else:
            result[field] = value
    return result


def od_collate_fn(batch):
    """
    batch is list of (image_tensor, coco_annots_list)
    coco_annots_list: list of dicts with keys 'bbox' and 'category_id'
    """
    images, boxes, labels, scores = [], [], [], []
    for img, annots in batch:
        images.append(img)
        # turn COCO bboxes [x, y, w, h] → [x1, y1, x2, y2]
        img_boxes, img_labels, img_scores = [], [], []
        for obj in annots:
            if "bbox" not in obj or "category_id" not in obj:
                raise ValueError(f"Malformed annotation object: {obj}")
            x, y, w, h = obj["bbox"]
            img_boxes.append([x, y, x + w, y + h])
            img_labels.append(map_raw_to_contiguous(obj["category_id"]))
            img_scores.append(1.0)  # ground truth label
        if img_boxes:
            boxes.append(torch.tensor(img_boxes, dtype=torch.float32))
            scores.append(torch.tensor(img_scores, dtype=torch.float32))
            labels.append(torch.tensor(img_labels, dtype=torch.int64))
        else:
            # no objects in this image
            boxes.append(torch.zeros((0, 4), dtype=torch.float32))
            scores.append(torch.zeros((0,), dtype=torch.float32))
            labels.append(torch.zeros((0,), dtype=torch.int64))
    images = torch.stack(images, 0)  # (B, C, H, W)
    return images, {"boxes": boxes, "labels": labels, "scores": scores}


class DataLoaderFactory:
    """
    The DataLoaderFactory class that creates a DataLoader for the given dataset.

    Attributes:
        None
    """

    @staticmethod
    def create_dataloader(
        config: Optional[DataLoaderConfig] = None,
        *,
        collate_fn: Callable | None = None,
        **kwargs,
        # config: Optional[DataLoaderConfig] = None, **kwargs
    ) -> TorchDataLoader:
        """
        A static method that creates a DataLoader for the given dataset with the given parameters.

        Args:
            config (DataLoaderConfig): The DataLoader configuration.
            **kwargs: Arbitrary keyword arguments for the DataLoader.

        Returns:
            TorchDataLoader: The DataLoader for the given dataset.

        Raises:
            ValueError: If the dataset is not of type TorchDataset.

        Note:
            It is possible to create a DataLoader without providing a DataLoaderConfig. In this case, the DataLoader will be created with the provided keyword arguments.
            DataLoaderConfig contains the following fields:
                - dataset: TorchDataset
                - batch_size: int
                - num_workers: int
                - shuffle: bool
                - drop_last: bool
                - pin_memory: bool
                - sampler: Optional[torch.utils.data.Sampler]

        """
        if config is None:
            # if no config is provided then create a new config based on the kwargs
            config = DataLoaderConfig(**kwargs)

        if not isinstance(config.dataset, TorchDataset):
            raise ValueError("Invalid dataset type provided. Expected TorchDataset.")

        if config.sampler is not None and config.shuffle:
            config.shuffle = False

        config_dict = dataclass_to_dict(config)
        # merge the config and kwargs
        params = {**config_dict, **kwargs}
        # PATCH: check for collate_fn as attribute if not explicitly passed
        if collate_fn is None and hasattr(config, "collate_fn"):
            collate_fn = getattr(config, "collate_fn")
        if collate_fn is not None:
            params["collate_fn"] = collate_fn
        dataloader = TorchDataLoader(**params)

        return dataloader
    
    @staticmethod
    def create_od_dataloader(
        config: Optional[DataLoaderConfig] = None, *, collate_fn=None, **kwargs
    ) -> TorchDataLoader:
        """
        Like create_dataloader, but defaults to the COCO object-detection collate.
        You can pass in either a DataLoaderConfig or the same kwargs you'd pass to
        create_dataloader. Any explicit kwarg here overrides the config.
        """
        if config is None:
            config = DataLoaderConfig(**kwargs)
        fn = collate_fn or od_collate_fn
        return DataLoaderFactory.create_dataloader(
            config=config, collate_fn=fn, **kwargs
        )
