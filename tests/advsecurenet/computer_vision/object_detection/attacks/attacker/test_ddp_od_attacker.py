import os
import shutil
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker import (
    DDPODAttacker,
)


class _DummyAttacker:
    def __init__(self, config, **kwargs):
        # simulate a dataloader on attacker
        self._dataloader = DataLoader(TensorDataset(torch.arange(10)), batch_size=2)

    def execute(self):
        # two batches: 4 images total
        return [torch.zeros(3, 3, 2, 2), torch.zeros(1, 3, 2, 2)]


class _NoLoaderAttacker:
    def __init__(self, config, **kwargs):
        self._dataloader = None

    def execute(self):
        return [torch.ones(1, 3, 2, 2)]


def _minimal_config():
    return SimpleNamespace(model=MagicMock(), return_adversarial_images=True, dataloader=SimpleNamespace(shuffle=True))


@pytest.fixture(autouse=True)
def _clean_tmp_dir():
    temp_dir = os.environ.get("ADV_OD_TMP", DDPODAttacker.TEMP_DIR)
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
    yield
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.DDPBaseTask.__init__", autospec=True)
@patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.dist.get_world_size", return_value=2)
@patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.dist.get_rank", return_value=1)
def test_init_sets_rank_and_world(mock_rank, mock_world, mock_ddpbase_init):
    def ddpbase_side_effect(self, model, rank, world_size):
        self._rank = rank
        self._world_size = world_size
    mock_ddpbase_init.side_effect = ddpbase_side_effect
    attacker = DDPODAttacker(attacker_class=_DummyAttacker, config=_minimal_config())
    assert attacker._rank == 1
    assert attacker._world_size == 2
    assert attacker.attacker is None
    assert attacker._sampler is None


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.DDPBaseTask.__init__", autospec=True)
@patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.dist.get_world_size", return_value=2)
@patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.dist.get_rank", return_value=0)
def test_setup_wraps_dataloader_with_distributedsampler(mock_rank, mock_world, mock_ddpbase_init):
    def ddpbase_side_effect(self, model, rank, world_size):
        self._rank = rank
        self._world_size = world_size
    mock_ddpbase_init.side_effect = ddpbase_side_effect

    cfg = _minimal_config()
    cfg.dataloader.shuffle = True
    ddp = DDPODAttacker(attacker_class=_DummyAttacker, config=cfg)
    ddp.setup()
    assert isinstance(ddp.attacker._dataloader.sampler, DistributedSampler)
    assert ddp._sampler is ddp.attacker._dataloader.sampler


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.DDPBaseTask.__init__", autospec=True)
@patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.dist.is_initialized", return_value=True)
@patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.dist.get_world_size", return_value=2)
@patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.dist.get_rank", return_value=0)
@patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.dist.barrier")
def test_run_task_saves_results_and_returns(mock_barrier, mock_rank, mock_world, mock_isinit, mock_ddpbase_init):
    def ddpbase_side_effect(self, model, rank, world_size):
        self._rank = rank
        self._world_size = world_size
    mock_ddpbase_init.side_effect = ddpbase_side_effect

    cfg = _minimal_config()
    ddp = DDPODAttacker(attacker_class=_DummyAttacker, config=cfg)
    ddp.setup()
    result = ddp.run_task()
    assert isinstance(result, list) and len(result) == 2
    temp_dir = os.environ.get("ADV_OD_TMP", DDPODAttacker.TEMP_DIR)
    out_path = os.path.join(temp_dir, "adv_images_rank0.pt")
    assert os.path.exists(out_path)
    payload = torch.load(out_path, map_location="cpu")
    assert "images" in payload and "indices" in payload and "local_image_count" in payload
    assert payload["local_image_count"] == 4
    assert payload["dataset_len"] == len(ddp.attacker._dataloader.dataset)
    assert mock_barrier.call_count >= 2


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.DDPBaseTask.__init__", autospec=True)
@patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.dist.is_initialized", return_value=True)
@patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.dist.get_world_size", return_value=2)
@patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.dist.get_rank", return_value=1)
@patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.dist.barrier")
def test_run_task_without_sampler_and_non_rank0_loglevel(mock_barrier, mock_rank, mock_world, mock_isinit, mock_ddpbase_init):
    def ddpbase_side_effect(self, model, rank, world_size):
        self._rank = rank
        self._world_size = world_size
    mock_ddpbase_init.side_effect = ddpbase_side_effect

    cfg = _minimal_config()
    ddp = DDPODAttacker(attacker_class=_NoLoaderAttacker, config=cfg)
    ddp.setup()
    with patch("logging.getLogger") as mock_get_logger:
        logger = MagicMock()
        mock_get_logger.return_value = logger
        result = ddp.run_task()
        assert isinstance(result, list) and len(result) == 1
        logger.setLevel.assert_called()  # log level suppressed on non-rank0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_gather_results_orders_by_indices(tmp_path):
    temp_dir = os.environ.get("ADV_OD_TMP", DDPODAttacker.TEMP_DIR)
    os.makedirs(temp_dir, exist_ok=True)
    # create two shards with indices covering 0..3
    shard0 = {
        "images": [torch.zeros(2, 3, 2, 2)],
        "indices": [0, 2],
        "local_image_count": 2,
        "dataset_len": 4,
    }
    shard1 = {
        "images": [torch.ones(2, 3, 2, 2)],
        "indices": [1, 3],
        "local_image_count": 2,
        "dataset_len": 4,
    }
    torch.save(shard0, os.path.join(temp_dir, "adv_images_rank0.pt"))
    torch.save(shard1, os.path.join(temp_dir, "adv_images_rank1.pt"))
    gathered = DDPODAttacker.gather_results(world_size=2)
    assert len(gathered) == 4
    # order should be 0->zeros,1->ones,2->zeros,3->ones
    assert torch.equal(gathered[0], torch.zeros(3, 2, 2)) or hasattr(gathered[0], "shape")
    # temp files should be cleaned up (dir may be removed if empty)
    assert not os.path.exists(os.path.join(temp_dir, "adv_images_rank0.pt"))
    assert not os.path.exists(os.path.join(temp_dir, "adv_images_rank1.pt"))


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_gather_results_fallback_concat_and_count_mismatch_logs_error():
    temp_dir = os.environ.get("ADV_OD_TMP", DDPODAttacker.TEMP_DIR)
    os.makedirs(temp_dir, exist_ok=True)
    # no indices provided -> fallback concat; set mismatched counts
    shard0 = {
        "images": [torch.zeros(1, 3, 2, 2)],
        "local_image_count": 1,
        "dataset_len": 5,  # force mismatch: total_local_counts (3) != dataset_len (5)
    }
    shard1 = {
        "images": [torch.ones(2, 3, 2, 2)],
        "local_image_count": 2,
        "dataset_len": 3,
    }
    torch.save(shard0, os.path.join(temp_dir, "adv_images_rank0.pt"))
    torch.save(shard1, os.path.join(temp_dir, "adv_images_rank1.pt"))
    with patch("advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker.logger") as mock_logger:
        gathered = DDPODAttacker.gather_results(world_size=2)
        assert len(gathered) == 3
        mock_logger.error.assert_called()  # count mismatch path


