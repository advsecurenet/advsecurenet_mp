import logging
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import Subset, TensorDataset

from advsecurenet.computer_vision.object_detection.attacks.pixel_perturbation_based.tog.tog_attack_type import TOGAttackType
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
@patch("cli.logic.attack.od_attacker.get_datasets")
@patch("cli.logic.attack.od_attacker.DDPCoordinator")
@patch("cli.logic.attack.od_attacker.DDPODAttacker")
@patch("cli.logic.attack.od_attacker.set_visible_gpus")
def test_execute_ddp_flow(mock_set_visible_gpus, mock_ddp_od_attacker, mock_ddp_coord, mock_get_datasets, od_attacker_config):
    class DummyAttackType:
        name = "DPATCH"

    # Simulate DDP path
    od_attacker_config.device = MagicMock()
    od_attacker_config.device.use_ddp = True
    od_attacker_config.device.gpu_ids = [0, 1]
    od_attacker_config.attack_procedure.save_result_images = True
    mock_ddp_coord.return_value.run.return_value = None
    mock_ddp_od_attacker.gather_results.return_value = ["imgA", "imgB"]
    mock_get_datasets.return_value = (MagicMock(), MagicMock())

    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    with patch("click.secho"):
        attacker.execute()
    mock_set_visible_gpus.assert_called_once()
    mock_ddp_coord.assert_called_once()
    mock_ddp_od_attacker.gather_results.assert_called_once_with(2)


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


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.get_datasets")
@patch("cli.logic.attack.od_attacker.create_model")
@patch("cli.logic.attack.od_attacker.get_object_detector")
@patch("cli.logic.attack.od_attacker.DPatch")
def test_prepare_attack_config_builds_configs(mock_dpatch, mock_get_object_detector, mock_create_model, mock_get_datasets, od_attacker_config):
    class DummyAttackType:
        name = "DPATCH"

    # create_model().model should have object_detector_config
    model_instance = MagicMock()
    model_instance.object_detector_config = {"foo": "bar"}
    mock_create_model.return_value = MagicMock(model=model_instance)
    mock_get_object_detector.return_value = MagicMock()
    mock_dpatch.return_value = MagicMock()
    mock_get_datasets.return_value = (MagicMock(), MagicMock())

    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    cfg, extras = attacker._prepare_attack_config()
    assert hasattr(cfg, "dataloader") and hasattr(cfg, "attack")
    mock_get_object_detector.assert_called_once()
    mock_dpatch.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.get_datasets")
def test_create_dataloader_config_sets_collate_for_coco(mock_get_datasets, od_attacker_config):
    class DummyAttackType:
        name = "DPATCH"
    mock_get_datasets.return_value = (MagicMock(), MagicMock())
    od_attacker_config.dataset.dataset_name = "COCO"
    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    dl_cfg = attacker._create_dataloader_config()
    assert hasattr(dl_cfg, "collate_fn")


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.get_datasets")
def test_create_dataloader_config_without_coco(mock_get_datasets, od_attacker_config):
    mock_get_datasets.return_value = (MagicMock(), MagicMock())

    class DummyAttackType:
        name = "DPATCH"

    od_attacker_config.dataset.dataset_name = "custom"
    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    dl_cfg = attacker._create_dataloader_config()
    assert not hasattr(dl_cfg, "collate_fn")


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.get_datasets")
def test_build_concrete_attacker_class_unknown_raises(mock_get_datasets, od_attacker_config):
    class DummyAttackType:
        name = "UNKNOWN"
    mock_get_datasets.return_value = (MagicMock(), MagicMock())
    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    with pytest.raises(ValueError):
        attacker._build_concrete_attacker_class()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.save_images")
@patch("cli.logic.attack.od_attacker.get_datasets", return_value=(MagicMock(), MagicMock()))
def test_save_images_if_needed_skips_when_no_images(
    mock_get_datasets, mock_save_images, od_attacker_config
):
    class DummyAttackType:
        name = "DPATCH"

    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    attacker._save_images_if_needed([])
    attacker._config.attack_procedure.save_result_images = False
    attacker._save_images_if_needed(["img"])
    mock_save_images.assert_not_called()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.torch.cuda.device_count", return_value=2)
@patch("cli.logic.attack.od_attacker.set_visible_gpus")
@patch("cli.logic.attack.od_attacker.DDPODAttacker")
@patch("cli.logic.attack.od_attacker.DDPCoordinator")
@patch("cli.logic.attack.od_attacker.get_datasets")
def test_execute_ddp_attack_populates_gpu_ids_and_handles_error(
    mock_get_datasets,
    mock_ddp_coordinator,
    mock_ddp_attacker,
    mock_set_visible,
    mock_device_count,
    od_attacker_config,
):
    class DummyAttackType:
        name = "DPATCH"

    od_attacker_config.device = MagicMock()
    od_attacker_config.device.use_ddp = True
    od_attacker_config.device.gpu_ids = []
    od_attacker_config.attack_procedure.save_result_images = True
    mock_get_datasets.return_value = (MagicMock(), MagicMock())
    mock_ddp_coordinator.return_value.run.return_value = None
    mock_ddp_attacker.gather_results.side_effect = RuntimeError("gather fail")

    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    with patch("cli.logic.attack.od_attacker.click.secho"):
        with patch.object(attacker, "_prepare_attack_config", return_value=(MagicMock(), {})):
            with patch.object(attacker, "_save_images_if_needed") as mock_save_images:
                attacker.execute()

    assert od_attacker_config.device.gpu_ids == [0, 1]
    mock_device_count.assert_called_once()
    mock_set_visible.assert_called_once_with([0, 1])
    mock_ddp_attacker.gather_results.assert_called_once_with(2)
    mock_save_images.assert_not_called()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.DDPODAttacker")
@patch("cli.logic.attack.od_attacker.CLIODAttacker._prepare_attack_config")
@patch("cli.logic.attack.od_attacker.torch.cuda.set_device")
@patch("cli.logic.attack.od_attacker.get_datasets")
def test_ddp_attack_fn_without_gpu_ids_sets_processor(
    mock_get_datasets,
    mock_set_device,
    mock_prepare,
    mock_ddp_wrapper,
    od_attacker_config,
):
    class DummyAttackType:
        name = "DPATCH"

    mock_get_datasets.return_value = (MagicMock(), MagicMock())
    od_attacker_config.device = MagicMock()
    od_attacker_config.device.use_ddp = True
    od_attacker_config.device.gpu_ids = None
    od_attacker_config.device.processor = None
    mock_prepare.return_value = (MagicMock(), {})
    mock_ddp_wrapper.return_value.setup.return_value = None
    mock_ddp_wrapper.return_value.run_task.return_value = None

    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    attacker._ddp_attack_fn(rank=1, world_size=2)

    mock_set_device.assert_called_with(1)
    assert od_attacker_config.device.processor == "cuda:1"
    mock_ddp_wrapper.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.DDPODAttacker")
@patch("cli.logic.attack.od_attacker.CLIODAttacker._prepare_attack_config", return_value=(MagicMock(), {}))
@patch("cli.logic.attack.od_attacker.torch.cuda.set_device", side_effect=RuntimeError("set fail"))
@patch("cli.logic.attack.od_attacker.get_datasets")
def test_ddp_attack_fn_logs_error_on_device_failure(
    mock_get_datasets,
    mock_set_device,
    mock_prepare,
    mock_ddp_wrapper,
    od_attacker_config,
    caplog,
):
    class DummyAttackType:
        name = "DPATCH"

    mock_get_datasets.return_value = (MagicMock(), MagicMock())
    od_attacker_config.device = MagicMock()
    od_attacker_config.device.use_ddp = True
    od_attacker_config.device.gpu_ids = [0]
    caplog.set_level(logging.ERROR, logger="cli.logic.attack.od_attacker")

    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    attacker._ddp_attack_fn(rank=0, world_size=1)

    assert "Failed to set device" in caplog.text
    mock_ddp_wrapper.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.get_object_detector")
@patch("cli.logic.attack.od_attacker.TOG")
@patch("cli.logic.attack.od_attacker.create_model")
@patch("cli.logic.attack.od_attacker.get_datasets")
def test_prepare_attack_config_tog_sets_extra_kwargs(
    mock_get_datasets,
    mock_create_model,
    mock_tog,
    mock_get_object_detector,
    od_attacker_config,
):
    class DummyAttackType:
        name = "TOG"

    mock_get_datasets.return_value = (MagicMock(), MagicMock())
    od_attacker_config.device = MagicMock()
    od_attacker_config.device.processor = "cuda:7"
    od_attacker_config.attack_config.attack_parameters.attack_type = "mislabeling"
    od_attacker_config.attack_config.attack_parameters.mislabeling_mode = "aggressive"
    od_attacker_config.attack_procedure.verbose = False

    model_instance = MagicMock()
    model_instance.object_detector_config = {"foo": "bar"}
    model_instance.to.return_value = model_instance
    mock_create_model.return_value = MagicMock(model=model_instance)
    mock_get_object_detector.return_value = MagicMock()

    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    cfg, extras = attacker._prepare_attack_config()

    mock_tog.assert_called_once()
    mock_get_object_detector.assert_called_once()
    model_instance.to.assert_called_once_with("cuda:7")
    assert extras["attack_type"] == TOGAttackType.MISLABELING
    assert extras["tog_mislabeling_mode"] == "aggressive"
    assert od_attacker_config.attack_config.attack_parameters.verbose is False
    assert cfg.device is od_attacker_config.device


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.get_datasets")
def test_sample_data_warns_when_sample_larger(mock_get_datasets, caplog, od_attacker_config):
    dataset = TensorDataset(torch.arange(6))
    mock_get_datasets.return_value = (dataset, MagicMock())

    class DummyAttackType:
        name = "DPATCH"

    od_attacker_config.dataset.random_sample_size = 10
    caplog.set_level(logging.WARNING, logger="cli.logic.attack.od_attacker")

    attacker = CLIODAttacker(od_attacker_config, DummyAttackType())
    subset = attacker._sample_data(dataset, 10)

    assert isinstance(subset, Subset)
    assert len(subset) == len(dataset)
    assert "smaller than the requested sample size" in caplog.text


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.attack.od_attacker.get_datasets")
def test_get_evaluators_override(mock_get_datasets, od_attacker_config):
    mock_get_datasets.return_value = (MagicMock(), MagicMock())

    class DummyAttackType:
        name = "DPATCH"

    attacker = CLIODAttacker(od_attacker_config, DummyAttackType(), evaluators=["foo"])
    assert attacker._get_evaluators() == ["foo"]
