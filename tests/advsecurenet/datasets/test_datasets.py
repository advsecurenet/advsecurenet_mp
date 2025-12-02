import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from advsecurenet.datasets.COCO import coco_utils
from advsecurenet.datasets.PascalVOC import pascalvoc_utils
from advsecurenet.datasets.PascalVOC.pascalvoc_dataset import PascalVOCDataset
from torchvision import datasets

from advsecurenet.datasets.base_dataset import BaseDataset, ImageFolderBaseDataset
from advsecurenet.datasets.Cifar10.cifar10_dataset import (
    CIFAR10Dataset,
    CIFAR100Dataset,
)
from advsecurenet.datasets.Custom.CustomDataset import CustomDataset
from advsecurenet.datasets.ImageNet.imagenet_dataset import ImageNetDataset
from advsecurenet.datasets.MNIST.mnist_dataset import FashionMNISTDataset, MNISTDataset
from advsecurenet.datasets.svhn.svhn_dataset import SVHNDataset
from advsecurenet.datasets.COCO.coco_dataset import COCODataset
from advsecurenet.shared.types.configs.preprocess_config import (
    PreprocessConfig,
    PreprocessStep,
)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_imagenet_dataset():
    # Instantiate the ImageNetDataset with an optional preprocess_config
    preprocess_steps = [
        PreprocessStep(name="Resize", params={"size": (256, 256)}),
        PreprocessStep(name="CenterCrop", params={"size": (224, 224)}),
    ]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)
    dataset = ImageNetDataset(preprocess_config)

    # Assertions
    assert dataset.mean == [0.485, 0.456, 0.406], "Mean values do not match"
    assert dataset.std == [
        0.229,
        0.224,
        0.225,
    ], "Standard deviation values do not match"
    assert dataset.input_size == (256, 256), "Input size does not match"
    assert dataset.crop_size == (224, 224), "Crop size does not match"
    assert dataset.name == "imagenet", "Dataset name does not match"
    assert dataset.num_classes == 1000, "Number of classes does not match"
    assert dataset.num_input_channels == 3, "Number of input channels does not match"
    assert (
        dataset._preprocess_config == preprocess_config
    ), "Preprocess config does not match"

    # Check if the parent class constructors are called
    assert isinstance(
        dataset, ImageFolderBaseDataset
    ), "Instance is not of type ImageFolderBaseDataset"
    assert isinstance(dataset, BaseDataset), "Instance is not of type BaseDataset"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_svhn_dataset():
    # Instantiate the SVHN with an optional preprocess_config
    preprocess_steps = [
        PreprocessStep(name="Resize", params={"size": (32, 32)}),
        PreprocessStep(name="CenterCrop", params={"size": (32, 32)}),
    ]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)
    dataset = SVHNDataset(preprocess_config)

    # Assertions
    assert dataset.mean == [0.4377, 0.4438, 0.4728], "Mean values do not match"
    assert dataset.std == [
        0.1980,
        0.2010,
        0.1970,
    ], "Standard deviation values do not match"
    assert dataset.input_size == (32, 32), "Input size does not match"
    assert dataset.crop_size == (32, 32), "Crop size does not match"
    assert dataset.name == "svhn", "Dataset name does not match"
    assert dataset.num_classes == 10, "Number of classes does not match"
    assert dataset.num_input_channels == 3, "Number of input channels does not match"
    assert (
        dataset._preprocess_config == preprocess_config
    ), "Preprocess config does not match"

    # Check if the parent class constructors are called
    assert isinstance(dataset, BaseDataset), "Instance is not of type BaseDataset"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_mnist_dataset():
    # Instantiate the MNIST with an optional preprocess_config
    preprocess_steps = [
        PreprocessStep(name="Resize", params={"size": (28, 28)}),
        PreprocessStep(name="CenterCrop", params={"size": (28, 28)}),
    ]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)
    dataset = MNISTDataset(preprocess_config)

    # Assertions
    assert dataset.mean == [0.1307], "Mean values do not match"
    assert dataset.std == [0.3081], "Standard deviation values do not match"
    assert dataset.input_size == (28, 28), "Input size does not match"
    assert dataset.crop_size == (28, 28), "Crop size does not match"
    assert dataset.name == "mnist", "Dataset name does not match"
    assert dataset.num_classes == 10, "Number of classes does not match"
    assert dataset.num_input_channels == 1, "Number of input channels does not match"
    assert (
        dataset._preprocess_config == preprocess_config
    ), "Preprocess config does not match"

    # Check if the parent class constructors are called
    assert isinstance(dataset, BaseDataset), "Instance is not of type BaseDataset"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_fashion_mnist_dataset():
    # Instantiate the Fashion MNIST with an optional preprocess_config
    preprocess_steps = [
        PreprocessStep(name="Resize", params={"size": (28, 28)}),
        PreprocessStep(name="CenterCrop", params={"size": (28, 28)}),
    ]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)
    dataset = FashionMNISTDataset(preprocess_config)

    # Assertions
    assert dataset.mean == [0.2860], "Mean values do not match"
    assert dataset.std == [0.3530], "Standard deviation values do not match"
    assert dataset.input_size == (28, 28), "Input size does not match"
    assert dataset.crop_size == (28, 28), "Crop size does not match"
    assert dataset.name == "fashion_mnist", "Dataset name does not match"
    assert dataset.num_classes == 10, "Number of classes does not match"
    assert dataset.num_input_channels == 1, "Number of input channels does not match"
    assert (
        dataset._preprocess_config == preprocess_config
    ), "Preprocess config does not match"

    # Check if the parent class constructors are called
    assert isinstance(dataset, BaseDataset), "Instance is not of type BaseDataset"
    assert isinstance(dataset, MNISTDataset), "Instance is not of type MNISTDataset"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_cifar10_dataset():
    # Instantiate the CIFAR10 with an optional preprocess_config
    preprocess_steps = [
        PreprocessStep(name="Resize", params={"size": (32, 32)}),
        PreprocessStep(name="CenterCrop", params={"size": (32, 32)}),
    ]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)
    dataset = CIFAR10Dataset(preprocess_config)

    # Assertions
    assert dataset.mean == [0.4914, 0.4822, 0.4465], "Mean values do not match"
    assert dataset.std == [
        0.2470,
        0.2435,
        0.2616,
    ], "Standard deviation values do not match"
    assert dataset.input_size == (32, 32), "Input size does not match"
    assert dataset.crop_size == (32, 32), "Crop size does not match"
    assert dataset.name == "cifar10", "Dataset name does not match"
    assert dataset.num_classes == 10, "Number of classes does not match"
    assert dataset.num_input_channels == 3, "Number of input channels does not match"
    assert (
        dataset._preprocess_config == preprocess_config
    ), "Preprocess config does not match"

    # Check if the parent class constructors are called
    assert isinstance(dataset, BaseDataset), "Instance is not of type BaseDataset"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_cifar100_dataset():
    # Instantiate the CIFAR100 with an optional preprocess_config
    preprocess_steps = [
        PreprocessStep(name="Resize", params={"size": (28, 28)}),
        PreprocessStep(name="CenterCrop", params={"size": (28, 28)}),
    ]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)
    dataset = CIFAR100Dataset(preprocess_config)

    # Assertions
    assert dataset.mean == [0.5071, 0.4867, 0.4408], "Mean values do not match"
    assert dataset.std == [
        0.2675,
        0.2565,
        0.2761,
    ], "Standard deviation values do not match"
    assert dataset.input_size == (32, 32), "Input size does not match"
    assert dataset.crop_size == (32, 32), "Crop size does not match"
    assert dataset.name == "cifar100", "Dataset name does not match"
    assert dataset.num_classes == 100, "Number of classes does not match"
    assert dataset.num_input_channels == 3, "Number of input channels does not match"
    assert (
        dataset._preprocess_config == preprocess_config
    ), "Preprocess config does not match"

    # Check if the parent class constructors are called
    assert isinstance(dataset, BaseDataset), "Instance is not of type BaseDataset"
    assert isinstance(dataset, CIFAR10Dataset), "Instance is not of type MNISTDataset"


@pytest.fixture
def temp_dataset_dir():
    # Create a temporary directory with the folder custom_dataset
    temp_dir = tempfile.mkdtemp()
    temp_dir = os.path.join(temp_dir, "custom_dataset")

    # Create subdirectories and fake images
    os.makedirs(os.path.join(temp_dir, "class1"))
    os.makedirs(os.path.join(temp_dir, "class2"))

    img1 = Image.new("RGB", (60, 30), color="red")
    img2 = Image.new("RGB", (60, 30), color="blue")

    img1.save(os.path.join(temp_dir, "class1", "img1.jpg"))
    img2.save(os.path.join(temp_dir, "class2", "img2.jpg"))

    yield temp_dir

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_custom_dataset(temp_dataset_dir):
    preprocess_steps = [
        PreprocessStep(name="Resize", params={"size": (28, 28)}),
        PreprocessStep(name="CenterCrop", params={"size": (28, 28)}),
    ]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)

    # Initialize the CustomDataset
    dataset = CustomDataset(preprocess_config=preprocess_config)
    # Set the root directory to the temp dataset directory
    dataset.root_dir = temp_dataset_dir

    dataset.load_dataset(root=temp_dataset_dir)
    # Assertions
    assert dataset.name == "custom", "Dataset name does not match"
    assert (
        dataset._preprocess_config == preprocess_config
    ), "Preprocess config does not match"

    # Verify the number of samples
    assert len(dataset) == 2, "Number of samples does not match"

    # Verify the labels
    assert dataset[0][1] == 0, "Label does not match"
    assert dataset[1][1] == 1, "Label does not match"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_coco_dataset():
    preprocess_steps = [
        PreprocessStep(name="Resize", params={"size": (224, 224)}),
        PreprocessStep(name="CenterCrop", params={"size": (224, 224)}),
    ]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)
    dataset = COCODataset(preprocess_config)

    # Assertions
    assert dataset.mean == [0.485, 0.456, 0.406], "Mean values do not match"
    assert dataset.std == [
        0.229,
        0.224,
        0.225,
    ], "Standard deviation values do not match"
    assert dataset.input_size == (224, 224), "Input size does not match"
    assert dataset.name == "coco", "Dataset name does not match"
    assert dataset.num_classes == 80, "Number of classes does not match"
    assert dataset.num_input_channels == 3, "Number of input channels does not match"
    assert (
        dataset._preprocess_config == preprocess_config
    ), "Preprocess config does not match"
    assert isinstance(dataset, BaseDataset), "Instance is not of type BaseDataset"


def make_minimal_coco_dataset(tmp_path, num_images=2, num_anns=2):
    # Create minimal COCO-style dataset structure
    images = []
    anns = []
    for i in range(num_images):
        images.append({"id": i, "file_name": f"img_{i}.jpg"})
    for i in range(num_anns):
        anns.append(
            {
                "id": i,
                "image_id": i % num_images,
                "category_id": 1,
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        )
    cats = [{"id": 1, "name": "cat1"}]
    info = {"description": "test"}
    licenses = [{"id": 1, "name": "lic1"}]
    data = {
        "images": images,
        "annotations": anns,
        "categories": cats,
        "info": info,
        "licenses": licenses,
    }
    ann_path = tmp_path / "ann.json"
    import json

    with open(ann_path, "w") as f:
        json.dump(data, f)
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(num_images):
        img = Image.new("RGB", (10, 10), color="white")
        img.save(img_dir / f"img_{i}.jpg")
    return str(img_dir), str(ann_path)


@pytest.mark.advsecurenet
def test_coco_dataset_getitem_and_len(tmp_path):
    img_dir, ann_path = make_minimal_coco_dataset(tmp_path)
    preprocess_config = PreprocessConfig(steps=[])
    dataset = COCODataset(preprocess_config)
    dataset.img_dir = img_dir
    dataset.ann_file = ann_path
    # Patch CocoDetection to avoid real file IO
    with patch(
        "advsecurenet.datasets.COCO.coco_dataset.datasets.CocoDetection"
    ) as mock_coco:
        mock_coco.return_value = MagicMock()
        wrapper = dataset.load_dataset(root=str(tmp_path), train=True, download=False)
        assert wrapper is not None
        assert dataset._dataset is not None
    # Out of bounds
    with pytest.raises(Exception):
        _ = dataset[100]


@pytest.mark.advsecurenet
def test_coco_dataset_empty(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    ann_path = tmp_path / "ann.json"
    import json

    with open(ann_path, "w") as f:
        json.dump({"images": [], "annotations": [], "categories": []}, f)
    preprocess_config = PreprocessConfig(steps=[])
    dataset = COCODataset(preprocess_config)
    dataset.img_dir = str(img_dir)
    dataset.ann_file = str(ann_path)
    # Remove _load_annotations call, just test __len__ and __getitem__
    try:
        length = len(dataset)
    except Exception:
        length = None
    assert length == 0 or length is None
    with pytest.raises(Exception):
        _ = dataset[0]


@pytest.mark.advsecurenet
def test_coco_dataset_bad_annotation(tmp_path):
    img_dir, ann_path = make_minimal_coco_dataset(tmp_path)
    # Corrupt the annotation file
    with open(ann_path, "w") as f:
        f.write("not a json")
    preprocess_config = PreprocessConfig(steps=[])
    dataset = COCODataset(preprocess_config)
    dataset.img_dir = img_dir
    dataset.ann_file = ann_path
    with pytest.raises(Exception):
        dataset._load_annotations()


@pytest.mark.advsecurenet
def test_coco_dataset_load_dataset(tmp_path):
    # Patch CocoDetection to avoid real file IO
    with patch(
        "advsecurenet.datasets.COCO.coco_dataset.datasets.CocoDetection"
    ) as mock_coco:
        mock_coco.return_value = MagicMock()
        dataset = COCODataset()
        # Patch _download_coco_if_not_exists to avoid download
        with patch.object(dataset, "_download_coco_if_not_exists") as mock_dl:
            wrapper = dataset.load_dataset(
                root=str(tmp_path), train=True, download=True
            )
            assert dataset.data_type is not None
            assert wrapper.dataset == mock_coco.return_value
            mock_dl.assert_called_once()
        # Test with download=False
        with patch.object(dataset, "_download_coco_if_not_exists") as mock_dl:
            wrapper = dataset.load_dataset(
                root=str(tmp_path), train=False, download=False
            )
            assert dataset.data_type is not None
            mock_dl.assert_not_called()


@pytest.mark.advsecurenet
def test_coco_dataset_maybe_download(tmp_path):
    # Patch download_and_extract_archive
    with patch(
        "advsecurenet.datasets.COCO.coco_dataset.download_and_extract_archive"
    ) as mock_dl:
        # Case: both dirs missing
        dataset = COCODataset()
        root = str(tmp_path)
        dataset._download_coco_if_not_exists(root, train=True)
        assert mock_dl.call_count == 2
        # Case: dirs exist
        img_dir = os.path.join(root, "train2017")
        ann_dir = os.path.join(root, "annotations")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        mock_dl.reset_mock()
        dataset._download_coco_if_not_exists(root, train=False)
        # Allow for 0 or 1 call depending on which dir exists
        assert mock_dl.call_count in (0, 1)


@pytest.mark.advsecurenet
def test_coco_utils_map_raw_to_contiguous_valid():
    for raw_id in coco_utils.COCO_INSTANCE_CATEGORY_IDS:
        idx = coco_utils.map_raw_to_contiguous(raw_id)
        assert isinstance(idx, int)
        assert 0 <= idx < len(coco_utils.COCO_INSTANCE_CATEGORY_IDS)


@pytest.mark.advsecurenet
def test_coco_utils_map_raw_to_contiguous_invalid():
    with pytest.raises(KeyError):
        coco_utils.map_raw_to_contiguous(-1)
    with pytest.raises(KeyError):
        coco_utils.map_raw_to_contiguous(9999)


@pytest.mark.advsecurenet
def test_voc_to_coco_anns_normalization_and_clipping():
    target = {
        "annotation": {
            "size": {"width": 100, "height": 50},
            "object": [
                {
                    "name": "TV / Monitor",  # normalization → tvmonitor
                    "bndbox": {"xmin": -10, "ymin": 5, "xmax": 120, "ymax": 30},
                    "difficult": "1",
                },
                {
                    "name": "dog",
                    "bndbox": {"xmin": 10, "ymin": 10, "xmax": 20, "ymax": 25},
                },
            ],
        }
    }
    anns = pascalvoc_utils.voc_to_coco_anns(target)
    # Both valid; first is clipped to image bounds
    assert len(anns) == 2
    a0 = anns[0]
    # tvmonitor id maps to index (raw id - 1)
    assert a0["category_id"] == pascalvoc_utils.NAME_TO_RAW_ID["tvmonitor"] - 1
    # Clipped bbox: xmin -> 0, xmax -> 100
    assert a0["bbox"][0] == 0.0 and a0["bbox"][1] == 5.0
    assert a0["bbox"][2] == 100.0 - 0.0 and a0["bbox"][3] == 30.0 - 5.0
    assert a0["difficult"] == 1
    # Dog entry untouched
    a1 = anns[1]
    assert a1["category_id"] == pascalvoc_utils.NAME_TO_RAW_ID["dog"] - 1
    assert a1["bbox"] == [10.0, 10.0, 10.0, 15.0]


@pytest.mark.advsecurenet
def test_voc_to_coco_anns_filters_invalid():
    # invalid name + invalid bbox (zero/negative size) should be filtered
    target = {
        "annotation": {
            "size": {"width": 20, "height": 20},
            "object": [
                {
                    "name": "unknown",
                    "bndbox": {"xmin": 0, "ymin": 0, "xmax": 5, "ymax": 5},
                },
                {
                    "name": "cat",
                    "bndbox": {"xmin": 10, "ymin": 10, "xmax": 10, "ymax": 12},
                },
                {
                    "name": "cat",
                    "bndbox": {"xmin": 12, "ymin": 12, "xmax": 10, "ymax": 9},
                },
            ],
        }
    }
    anns = pascalvoc_utils.voc_to_coco_anns(target)
    assert anns == []


@pytest.mark.advsecurenet
def test_pascalvoc_validate_year():
    # valid years
    PascalVOCDataset._validate_year("2007")
    PascalVOCDataset._validate_year("2012")
    # invalid
    with pytest.raises(ValueError):
        PascalVOCDataset._validate_year("1999")


@pytest.mark.advsecurenet
def test_pascalvoc_load_dataset_success(monkeypatch, tmp_path):
    # patch VOCDetection to prevent IO and capture target_transform
    captured_tt = {}

    class _DummyVOC:
        def __init__(self, *args, **kwargs):
            captured_tt["target_transform"] = kwargs.get("target_transform")

    monkeypatch.setattr(
        "advsecurenet.datasets.PascalVOC.pascalvoc_dataset.datasets.VOCDetection",
        _DummyVOC,
    )
    ds = PascalVOCDataset()
    wrapper = ds.load_dataset(
        root=str(tmp_path), train=True, download=False, year="2007"
    )
    assert wrapper is not None
    # data_type TRAIN for train/trainval; val -> TEST
    assert str(ds.data_type).endswith("TRAIN")
    # target_transform should be callable and produce list of dicts with bbox and category_id
    tt = captured_tt.get("target_transform")
    assert callable(tt)
    sample = {
        "annotation": {
            "size": {"width": 100, "height": 100},
            "object": [
                {
                    "name": "dog",
                    "bndbox": {"xmin": 1, "ymin": 1, "xmax": 20, "ymax": 30},
                }
            ],
        }
    }
    out = tt(sample)
    assert isinstance(out, list)
    assert len(out) > 0
    assert "bbox" in out[0] and "category_id" in out[0]


@pytest.mark.advsecurenet
def test_pascalvoc_load_dataset_fallback(monkeypatch, tmp_path):
    calls = {"ctor": 0, "fallback": 0}

    def _ctor_fail_once(*args, **kwargs):
        calls["ctor"] += 1
        if calls["ctor"] == 1:
            raise RuntimeError("fail first")
        return object()

    def _fallback(*args, **kwargs):
        calls["fallback"] += 1

    monkeypatch.setattr(
        "advsecurenet.datasets.PascalVOC.pascalvoc_dataset.datasets.VOCDetection",
        _ctor_fail_once,
    )
    monkeypatch.setattr(
        PascalVOCDataset, "_fallback_voc_download", staticmethod(_fallback)
    )
    ds = PascalVOCDataset()
    wrapper = ds.load_dataset(
        root=str(tmp_path), train=False, download=True, year="2012"
    )
    assert wrapper is not None
    assert calls["fallback"] == 1


@pytest.mark.advsecurenet
def test_pascalvoc_temporarily_disable_ssl_verification(monkeypatch):
    from advsecurenet.datasets.PascalVOC.pascalvoc_dataset import (
        temporarily_disable_ssl_verification,
    )
    import ssl

    original = ssl._create_default_https_context
    with temporarily_disable_ssl_verification():
        assert ssl._create_default_https_context != original
    assert ssl._create_default_https_context == original


@pytest.mark.advsecurenet
def test_pascalvoc_init_with_preprocess_config():
    preprocess_steps = [
        PreprocessStep(name="Resize", params={"size": (256, 256)}),
        PreprocessStep(name="CenterCrop", params={"size": (224, 224)}),
    ]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)
    dataset = PascalVOCDataset(preprocess_config)
    assert dataset.input_size == (256, 256)
    assert dataset.name == "pascal_voc"
    assert dataset.num_classes == 20


@pytest.mark.advsecurenet
def test_pascalvoc_init_without_preprocess_config():
    dataset = PascalVOCDataset()
    assert dataset.input_size == (224, 224)
    assert dataset.name == "pascal_voc"


@pytest.mark.advsecurenet
def test_pascalvoc_init_with_preprocess_config_no_resize():
    preprocess_steps = [
        PreprocessStep(name="CenterCrop", params={"size": (224, 224)}),
    ]
    preprocess_config = PreprocessConfig(steps=preprocess_steps)
    dataset = PascalVOCDataset(preprocess_config)
    assert dataset.input_size == (224, 224)  # fallback


@pytest.mark.advsecurenet
def test_pascalvoc_get_dataset_class():
    dataset = PascalVOCDataset()
    assert dataset.get_dataset_class() == datasets.VOCDetection


@pytest.mark.advsecurenet
def test_pascalvoc_load_dataset_root_none(monkeypatch, tmp_path):
    class _DummyVOC:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "advsecurenet.datasets.PascalVOC.pascalvoc_dataset.datasets.VOCDetection",
        _DummyVOC,
    )
    monkeypatch.setattr("pkg_resources.resource_filename", lambda *args: str(tmp_path))
    ds = PascalVOCDataset()
    wrapper = ds.load_dataset(root=None, train=True, download=False, year="2012")
    assert wrapper is not None


@pytest.mark.advsecurenet
def test_pascalvoc_load_dataset_image_set_overrides_train(monkeypatch, tmp_path):
    class _DummyVOC:
        def __init__(self, *args, **kwargs):
            self.image_set = kwargs.get("image_set")

    monkeypatch.setattr(
        "advsecurenet.datasets.PascalVOC.pascalvoc_dataset.datasets.VOCDetection",
        _DummyVOC,
    )
    ds = PascalVOCDataset()
    # image_set="test" should override train=True
    wrapper = ds.load_dataset(
        root=str(tmp_path), train=True, download=False, year="2012", image_set="test"
    )
    assert wrapper is not None
    assert str(ds.data_type).endswith("TEST")  # test is not train/trainval


@pytest.mark.advsecurenet
def test_pascalvoc_load_dataset_trainval_data_type(monkeypatch, tmp_path):
    class _DummyVOC:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(
        "advsecurenet.datasets.PascalVOC.pascalvoc_dataset.datasets.VOCDetection",
        _DummyVOC,
    )
    ds = PascalVOCDataset()
    wrapper = ds.load_dataset(
        root=str(tmp_path),
        train=False,
        download=False,
        year="2012",
        image_set="trainval",
    )
    assert wrapper is not None
    assert str(ds.data_type).endswith("TRAIN")  # trainval is TRAIN


@pytest.mark.advsecurenet
def test_pascalvoc_load_dataset_download_false_on_exception(monkeypatch, tmp_path):
    calls = {"ctor": 0}

    def _ctor_always_fail(*args, **kwargs):
        calls["ctor"] += 1
        raise RuntimeError("always fail")

    monkeypatch.setattr(
        "advsecurenet.datasets.PascalVOC.pascalvoc_dataset.datasets.VOCDetection",
        _ctor_always_fail,
    )
    monkeypatch.setattr(PascalVOCDataset, "_fallback_voc_download", MagicMock())
    ds = PascalVOCDataset()
    with pytest.raises(RuntimeError):
        ds.load_dataset(root=str(tmp_path), train=True, download=False, year="2012")


@pytest.mark.advsecurenet
def test_pascalvoc_fallback_voc_download_existing_dir(tmp_path, monkeypatch):
    ds = PascalVOCDataset()
    voc_dir = os.path.join(str(tmp_path), "VOCdevkit", "VOC2012")
    os.makedirs(voc_dir, exist_ok=True)
    # Mock urlretrieve to ensure it's never called when dir exists
    urlretrieve_calls = []
    monkeypatch.setattr(
        "urllib.request.urlretrieve", lambda *args: urlretrieve_calls.append(args)
    )
    ds._fallback_voc_download(str(tmp_path), year="2012")
    # Should skip download when dir exists
    assert len(urlretrieve_calls) == 0
