import logging
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import Subset, TensorDataset

from cli.logic.attack.od_attacker import CLIODAttacker

logger = logging.getLogger("cli.logic.attack.od_attacker")


@pytest.fixture
def od_attacker_config():
    config = MagicMock()
    config.device = "cpu"
    config.attack_procedure = MagicMock()
    config.attack_procedure.save_result_images = True
    config.attack_procedure.result_images_dir = "results"
    config.attack_procedure.result_images_prefix = "adv"
    config.attack_procedure.verbose = True
    config.dataloader = MagicMock()
    config.dataloader.default = MagicMock()
    config.dataloader.default.batch_size = 2
    config.dataloader.default.num_workers = 0
    config.dataloader.default.shuffle = False
    config.dataloader.default.drop_last = False
    config.dataloader.default.pin_memory = False
    config.dataset = MagicMock()
    config.dataset.load_splits = ["train"]
    config.dataset.random_sample_size = None
    config.dataset.dataset_name = "COCO"
    config.model = MagicMock()
    config.model.object_detector_config = {}
    config.attack_config = MagicMock()
    config.attack_config.attack_parameters = MagicMock()
    config.attack_config.attack_parameters.object_detector = "yolov5"
    config.attack_config.attack_parameters.attack_type = "vanishing"
    config.attack_config.attack_parameters.mislabeling_mode = "ml"
    config.attack_config.attack_parameters.verbose = True
    return config


@pytest.fixture
def od_attacker(od_attacker_config):
    class DummyAttackType:
        name = "DPATCH"

    return CLIODAttacker(od_attacker_config, DummyAttackType())


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.create_model")
@patch("cli.logic.attack.od_attacker.get_object_detector")
@patch("cli.logic.attack.od_attacker.DPatch")
@patch("cli.logic.attack.od_attacker.AdversarialPatchODAttacker")
@patch("cli.logic.attack.od_attacker.save_images")
@patch("cli.logic.attack.od_attacker.get_datasets")
def test_execute_dpatch(
    mock_get_datasets,
    mock_save_images,
    mock_adv_patch_attacker,
    mock_dpatch,
    mock_get_object_detector,
    mock_create_model,
    od_attacker_config,
):
    class DummyAttackType:
        name = "DPATCH"

    mock_adv_patch_attacker.return_value.execute.return_value = ["img1", "img2"]
    mock_get_datasets.return_value = (MagicMock(), MagicMock())
    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    with patch("click.secho") as mock_click_secho:
        attacker.execute()
    mock_adv_patch_attacker.return_value.execute.assert_called_once()
    mock_save_images.assert_called_once()
    mock_click_secho.assert_called_once_with(
        "Attack completed successfully.", fg="green"
    )


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.create_model")
@patch("cli.logic.attack.od_attacker.get_object_detector")
@patch("cli.logic.attack.od_attacker.TOG")
@patch("cli.logic.attack.od_attacker.PixelPerturbationODAttacker")
@patch("cli.logic.attack.od_attacker.save_images")
@patch("cli.logic.attack.od_attacker.get_datasets")
def test_execute_tog(
    mock_get_datasets,
    mock_save_images,
    mock_pixel_attacker,
    mock_tog,
    mock_get_object_detector,
    mock_create_model,
    od_attacker_config,
):
    class DummyAttackType:
        name = "TOG"

    mock_pixel_attacker.return_value.execute.return_value = ["img1"]
    mock_get_datasets.return_value = (MagicMock(), MagicMock())
    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    with patch("click.secho") as mock_click_secho:
        attacker.execute()
    mock_pixel_attacker.return_value.execute.assert_called_once()
    mock_save_images.assert_called_once()
    mock_click_secho.assert_called_once_with(
        "Attack completed successfully.", fg="green"
    )


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.create_model")
@patch("cli.logic.attack.od_attacker.get_object_detector")
@patch("cli.logic.attack.od_attacker.DPatch")
@patch("cli.logic.attack.od_attacker.AdversarialPatchODAttacker")
@patch("cli.logic.attack.od_attacker.save_images")
@patch("cli.logic.attack.od_attacker.get_datasets")
def test_execute_attack_unknown_type(
    mock_get_datasets,
    mock_save_images,
    mock_adv_patch_attacker,
    mock_dpatch,
    mock_get_object_detector,
    mock_create_model,
    od_attacker_config,
):
    class DummyAttackType:
        name = "UNKNOWN"

    mock_get_datasets.return_value = (MagicMock(), MagicMock())
    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    with pytest.raises(ValueError, match="Unknown attack type: <.*DummyAttackType.*>"):
        attacker._execute_attack()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.get_datasets")
def test_prepare_dataset_train_split(mock_get_datasets, od_attacker_config):
    mock_train = MagicMock()
    mock_test = MagicMock()
    mock_get_datasets.return_value = (mock_train, mock_test)

    class DummyAttackType:
        name = "DPATCH"

    od_attacker_config.dataset.load_splits = ["train"]
    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    assert attacker._dataset == mock_train


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.get_datasets")
def test_prepare_dataset_test_split(mock_get_datasets, od_attacker_config):
    mock_train = MagicMock()
    mock_test = MagicMock()
    mock_get_datasets.return_value = (mock_train, mock_test)

    class DummyAttackType:
        name = "DPATCH"

    od_attacker_config.dataset.load_splits = ["test"]
    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    assert attacker._dataset == mock_test


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.get_datasets")
def test_prepare_dataset_both_splits(mock_get_datasets, od_attacker_config):
    mock_train = MagicMock()
    mock_test = MagicMock()
    mock_get_datasets.return_value = (mock_train, mock_test)

    class DummyAttackType:
        name = "DPATCH"

    od_attacker_config.dataset.load_splits = ["train", "test"]
    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    assert attacker._dataset == mock_train + mock_test


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.random_split")
@patch("cli.logic.attack.od_attacker.get_datasets")
def test_sample_data_if_required_sampling(
    mock_get_datasets, mock_random_split, od_attacker_config
):
    class DummyAttackType:
        name = "DPATCH"

    od_attacker_config.dataset.random_sample_size = 2

    # Use a real TensorDataset
    real_dataset = torch.arange(5).unsqueeze(1)  # shape (5, 1)
    dataset = TensorDataset(real_dataset)
    real_indices = [0, 1]
    subset0 = Subset(dataset, real_indices)
    subset1 = Subset(dataset, [2, 3, 4])
    mock_get_datasets.return_value = (dataset, MagicMock())
    mock_random_split.return_value = [subset0, subset1]
    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    # Now test the public interface, which will call _sample_data_if_required internally
    sampled = attacker._sample_data_if_required(dataset)
    assert isinstance(sampled, Subset)
    assert list(sampled.indices) == real_indices


@pytest.mark.cli
@pytest.mark.essential
def test_sample_data_if_required_no_sampling(od_attacker_config):
    class DummyAttackType:
        name = "DPATCH"

    od_attacker_config.dataset.random_sample_size = None
    od_attacker_config.dataset.split_config = None
    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    mock_data = MagicMock()
    result = attacker._sample_data_if_required(mock_data)
    assert result == mock_data
