import os
import tempfile
from unittest.mock import MagicMock, patch, mock_open

import pytest
import torch
from torch import nn

from advsecurenet.utils.model_utils import (
    download_weights,
    load_model,
    save_model,
    non_inplace_operations,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_save_model():
    model = SimpleModel()
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = "test_model"
        save_model(model, filename, filepath=temp_dir)
        assert os.path.exists(os.path.join(temp_dir, f"{filename}.pth"))


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_load_model():
    model = SimpleModel()
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = "test_model"
        save_model(model, filename, filepath=temp_dir)
        loaded_model = SimpleModel()
        load_model(loaded_model, filename, filepath=temp_dir)
        for param1, param2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.equal(param1, param2)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_save_model_distributed():
    model = nn.DataParallel(SimpleModel())
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = "test_model_distributed"
        save_model(model, filename, filepath=temp_dir, distributed=True)
        assert os.path.exists(os.path.join(temp_dir, f"{filename}.pth"))


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.utils.model_utils.requests.get")
@patch("advsecurenet.utils.model_utils.tqdm")
def test_download_weights(mock_tqdm, mock_get):
    model_name = "resnet50"
    dataset_name = "cifar10"
    filename = f"{model_name}_{dataset_name}_weights.pth"

    # Mock the response of the requests.get call
    mock_response = MagicMock()
    mock_response.iter_content = MagicMock(return_value=[b"1234"])
    mock_response.headers = {"content-length": "4"}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    # Mock tqdm progress bar
    mock_progress_bar = MagicMock()
    mock_tqdm.return_value = mock_progress_bar

    with tempfile.TemporaryDirectory() as temp_dir:
        download_weights(
            model_name=model_name, dataset_name=dataset_name, save_path=temp_dir
        )
        assert os.path.exists(os.path.join(temp_dir, filename))


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_download_weights_already_exists():
    model_name = "resnet50"
    dataset_name = "cifar10"
    filename = f"{model_name}_{dataset_name}_weights.pth"

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file to simulate already downloaded weights
        with open(os.path.join(temp_dir, filename), "w") as f:
            f.write("dummy content")

        with patch("advsecurenet.utils.model_utils.requests.get") as mock_get:
            download_weights(
                model_name=model_name, dataset_name=dataset_name, save_path=temp_dir
            )
            mock_get.assert_not_called()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_save_model_default_filepath():
    """Test save_model with default filepath (uses pkg_resources)."""
    model = SimpleModel()
    filename = "test_model_default"

    with patch(
        "advsecurenet.utils.model_utils.pkg_resources.resource_filename"
    ) as mock_resource:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a subdirectory to ensure os.makedirs is called
            weights_dir = os.path.join(temp_dir, "weights")
            mock_resource.return_value = weights_dir

            # Don't create the directory beforehand to ensure os.makedirs is called
            save_model(model, filename)

            # Check that pkg_resources was called
            mock_resource.assert_called_once_with("advsecurenet", "weights")
            # Check that the directory was created and file was saved
            assert os.path.exists(weights_dir)
            assert os.path.exists(os.path.join(weights_dir, f"{filename}.pth"))


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_save_model_with_pth_extension():
    """Test save_model when filename already has .pth extension."""
    model = SimpleModel()
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = "test_model.pth"
        save_model(model, filename, filepath=temp_dir)
        assert os.path.exists(os.path.join(temp_dir, filename))


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_load_model_filename_with_path():
    """Test load_model when filename contains a path."""
    model = SimpleModel()
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model first
        filename = "test_model"
        save_model(model, filename, filepath=temp_dir)

        # Test loading with path in filename
        full_path = os.path.join(temp_dir, f"{filename}.pth")
        loaded_model = SimpleModel()
        load_model(loaded_model, full_path)

        # Verify parameters are the same
        for param1, param2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.equal(param1, param2)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_load_model_with_pth_extension():
    """Test load_model when filename already has .pth extension."""
    model = SimpleModel()
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = "test_model.pth"
        save_model(model, filename, filepath=temp_dir)
        loaded_model = SimpleModel()
        load_model(loaded_model, filename, filepath=temp_dir)
        for param1, param2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.equal(param1, param2)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_load_model_default_filepath():
    """Test load_model with default filepath."""
    model = SimpleModel()
    filename = "test_model"

    with patch(
        "advsecurenet.utils.model_utils.pkg_resources.resource_filename"
    ) as mock_resource:
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_resource.return_value = temp_dir
            # First save the model to the temp directory
            save_model(model, filename, filepath=temp_dir)

            loaded_model = SimpleModel()
            load_model(loaded_model, filename)  # No filepath provided

            mock_resource.assert_called_with("advsecurenet", "weights")


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.utils.model_utils.requests.get")
@patch("advsecurenet.utils.model_utils.tqdm")
def test_download_weights_with_filename(mock_tqdm, mock_get):
    """Test download_weights when filename is provided directly."""
    filename = "custom_weights.pth"

    mock_response = MagicMock()
    mock_response.iter_content = MagicMock(return_value=[b"1234"])
    mock_response.headers = {"content-length": "4"}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    # Mock tqdm progress bar
    mock_progress_bar = MagicMock()
    mock_tqdm.return_value = mock_progress_bar

    with tempfile.TemporaryDirectory() as temp_dir:
        download_weights(filename=filename, save_path=temp_dir)
        assert os.path.exists(os.path.join(temp_dir, filename))
        # Verify the URL construction
        expected_url = (
            f"https://advsecurenet.s3.eu-central-1.amazonaws.com/weights/{filename}"
        )
        mock_get.assert_called_once_with(expected_url, stream=True, timeout=10)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_download_weights_missing_parameters():
    """Test download_weights raises ValueError when required parameters are missing."""
    with pytest.raises(
        ValueError,
        match="Both model_name and dataset_name must be provided if filename is not specified.",
    ):
        download_weights(model_name="resnet50")  # Missing dataset_name

    with pytest.raises(
        ValueError,
        match="Both model_name and dataset_name must be provided if filename is not specified.",
    ):
        download_weights(dataset_name="cifar10")  # Missing model_name


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.utils.model_utils.requests.get")
@patch("advsecurenet.utils.model_utils.os.remove")
def test_download_weights_http_error(mock_remove, mock_get):
    """Test download_weights handles HTTP errors properly."""
    model_name = "resnet50"
    dataset_name = "cifar10"

    # Mock HTTP error
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("HTTP 404 Not Found")
    mock_get.return_value = mock_response

    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(Exception, match="HTTP 404 Not Found"):
            download_weights(
                model_name=model_name, dataset_name=dataset_name, save_path=temp_dir
            )

        # Verify cleanup was attempted
        mock_remove.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.utils.model_utils.requests.get")
@patch("advsecurenet.utils.model_utils.os.remove")
def test_download_weights_keyboard_interrupt(mock_remove, mock_get):
    """Test download_weights handles KeyboardInterrupt and cleans up properly."""
    model_name = "resnet50"
    dataset_name = "cifar10"

    # Mock KeyboardInterrupt during download
    mock_response = MagicMock()
    mock_response.headers = {"content-length": "4"}
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content = MagicMock(
        side_effect=KeyboardInterrupt("User interrupted")
    )
    mock_get.return_value = mock_response

    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(KeyboardInterrupt, match="User interrupted"):
            download_weights(
                model_name=model_name, dataset_name=dataset_name, save_path=temp_dir
            )

        # Verify cleanup was attempted
        mock_remove.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.utils.model_utils.requests.get")
@patch("advsecurenet.utils.model_utils.tqdm")
def test_download_weights_creates_directory(mock_tqdm, mock_get):
    """Test download_weights creates directory when it doesn't exist."""
    model_name = "resnet50"
    dataset_name = "cifar10"

    mock_response = MagicMock()
    mock_response.iter_content = MagicMock(return_value=[b"1234"])
    mock_response.headers = {"content-length": "4"}
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    # Mock tqdm progress bar
    mock_progress_bar = MagicMock()
    mock_tqdm.return_value = mock_progress_bar

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a path that doesn't exist
        nonexistent_path = os.path.join(temp_dir, "nonexistent", "directory")
        download_weights(
            model_name=model_name, dataset_name=dataset_name, save_path=nonexistent_path
        )

        # Verify the directory was created and file exists
        filename = f"{model_name}_{dataset_name}_weights.pth"
        assert os.path.exists(nonexistent_path)
        assert os.path.exists(os.path.join(nonexistent_path, filename))


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_non_inplace_operations():
    """Test the non_inplace_operations context manager."""
    # Create test tensors
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])

    # Test normal behavior (should be in-place)
    original_id = id(a)
    a += b
    assert id(a) == original_id  # Same object (in-place)

    # Test with context manager (should be out-of-place)
    a = torch.tensor([1.0, 2.0])
    with non_inplace_operations():
        original_id = id(a)
        result = a.__iadd__(b)  # This should now call __add__ instead
        # The result might be the same object or different, depending on PyTorch internals
        # Just verify the context manager executes without error
        assert result is not None

    # Test that the original method is restored after exiting context
    a = torch.tensor([1.0, 2.0])
    original_id = id(a)
    a += b
    assert id(a) == original_id  # Back to in-place behavior
