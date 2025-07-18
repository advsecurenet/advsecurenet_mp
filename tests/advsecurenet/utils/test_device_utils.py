import torch
import pytest
from advsecurenet.utils.device_utils import move_batch_to_device, setup_device
from unittest.mock import patch


def test_move_tensor_to_device():
    x = torch.zeros(2, 3)
    y = {'label': torch.ones(2)}
    device = torch.device('cpu')
    x_out, y_out = move_batch_to_device(x, y, device)
    assert x_out.device == device
    assert y_out['label'].device == device

def test_move_dict_to_device():
    x = {'img': torch.zeros(2, 3), 'meta': {'foo': torch.ones(1)}}
    y = {'label': torch.ones(2)}
    device = torch.device('cpu')
    x_out, y_out = move_batch_to_device(x, y, device)
    assert x_out['img'].device == device
    assert x_out['meta']['foo'].device == device
    assert y_out['label'].device == device

def test_move_list_to_device():
    x = [torch.zeros(2, 3), torch.ones(2, 3)]
    y = [{'label': torch.ones(2)}, {'label': torch.zeros(2)}]
    device = torch.device('cpu')
    x_out, y_out = move_batch_to_device(x, y, device)
    assert all(t.device == device for t in x_out)
    assert all(d['label'].device == device for d in y_out)

def test_move_nested_structure_to_device():
    x = {'imgs': [torch.zeros(1), torch.ones(1)], 'meta': {'foo': [torch.ones(1), 42]}}
    y = {'labels': [torch.ones(1), 'not_a_tensor']}
    device = torch.device('cpu')
    x_out, y_out = move_batch_to_device(x, y, device)
    assert all(t.device == device for t in x_out['imgs'])
    assert x_out['meta']['foo'][0].device == device
    assert x_out['meta']['foo'][1] == 42
    assert y_out['labels'][0].device == device
    assert y_out['labels'][1] == 'not_a_tensor'

def test_move_non_tensor_leaves():
    x = 123
    y = 'abc'
    device = torch.device('cpu')
    x_out, y_out = move_batch_to_device(x, y, device)
    assert x_out == 123
    assert y_out == 'abc'

# --- Tests for setup_device ---
def test_setup_device_cpu():
    device = setup_device('cpu')
    assert isinstance(device, torch.device)
    assert device.type == 'cpu'

def test_setup_device_cuda():
    device = setup_device('cuda')
    assert isinstance(device, torch.device)
    assert device.type == 'cuda'

def test_setup_device_none_cuda_available():
    with patch('torch.cuda.is_available', return_value=True):
        device = setup_device(None)
        assert isinstance(device, torch.device)
        assert device.type == 'cuda'

def test_setup_device_none_cuda_not_available():
    with patch('torch.cuda.is_available', return_value=False):
        device = setup_device(None)
        assert isinstance(device, torch.device)
        assert device.type == 'cpu'
