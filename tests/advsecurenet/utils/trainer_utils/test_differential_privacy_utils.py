"""
Tests for differential privacy utilities.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from advsecurenet.utils.trainer_utils.differential_privacy_utils import (
    setup_privacy_engine,
)
from shared.types.configs.base import DifferentialPrivacyBase


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def simple_model():
    """Fixture to create a simple model."""
    return SimpleModel()


@pytest.fixture
def optimizer(simple_model):
    """Fixture to create an optimizer."""
    return optim.SGD(simple_model.parameters(), lr=0.01)


@pytest.fixture
def data_loader():
    """Fixture to create a simple data loader."""
    # Create dummy data
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32)


@pytest.fixture
def dp_config():
    """Fixture to create a differential privacy configuration."""
    return DifferentialPrivacyBase(
        enable=True,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        delta=1e-5,
        kwargs=None,
    )


@pytest.fixture
def dp_config_with_fast_clipping():
    """Fixture to create a differential privacy configuration with fast clipping."""
    return DifferentialPrivacyBase(
        enable=True,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        delta=1e-5,
        kwargs={"clipping": "fast"},
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.utils.trainer_utils.differential_privacy_utils.PrivacyEngine")
def test_setup_privacy_engine_standard_clipping(
    mock_privacy_engine_class, simple_model, optimizer, data_loader, dp_config
):
    """Test setup_privacy_engine with standard clipping (returns 3 items)."""
    # Create a mock privacy engine instance
    mock_privacy_engine = MagicMock()
    mock_privacy_engine_class.return_value = mock_privacy_engine

    # Mock the make_private method to return 3 items (standard clipping)
    mock_private_model = MagicMock()
    mock_private_optimizer = MagicMock()
    mock_private_data_loader = MagicMock()
    mock_privacy_engine.make_private.return_value = (
        mock_private_model,
        mock_private_optimizer,
        mock_private_data_loader,
    )

    # Call the function
    result = setup_privacy_engine(simple_model, optimizer, data_loader, dp_config)

    # Verify the results
    assert len(result) == 5
    (
        private_model,
        private_optimizer,
        private_data_loader,
        privacy_engine,
        private_loss_fn,
    ) = result

    assert private_model == mock_private_model
    assert private_optimizer == mock_private_optimizer
    assert private_data_loader == mock_private_data_loader
    assert privacy_engine == mock_privacy_engine
    assert private_loss_fn is None  # Should be None for standard clipping

    # Verify make_private was called with correct parameters
    mock_privacy_engine.make_private.assert_called_once_with(
        module=simple_model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=dp_config.noise_multiplier,
        max_grad_norm=dp_config.max_grad_norm,
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.utils.trainer_utils.differential_privacy_utils.PrivacyEngine")
def test_setup_privacy_engine_fast_clipping(
    mock_privacy_engine_class,
    simple_model,
    optimizer,
    data_loader,
    dp_config_with_fast_clipping,
):
    """Test setup_privacy_engine with fast clipping (returns 4 items)."""
    # Create a mock privacy engine instance
    mock_privacy_engine = MagicMock()
    mock_privacy_engine_class.return_value = mock_privacy_engine

    # Mock the make_private method to return 4 items (fast clipping)
    mock_private_model = MagicMock()
    mock_private_optimizer = MagicMock()
    mock_private_loss_fn = MagicMock()
    mock_private_data_loader = MagicMock()
    mock_privacy_engine.make_private.return_value = (
        mock_private_model,
        mock_private_optimizer,
        mock_private_loss_fn,
        mock_private_data_loader,
    )

    # Call the function
    result = setup_privacy_engine(
        simple_model, optimizer, data_loader, dp_config_with_fast_clipping
    )

    # Verify the results
    assert len(result) == 5
    (
        private_model,
        private_optimizer,
        private_data_loader,
        privacy_engine,
        private_loss_fn,
    ) = result

    assert private_model == mock_private_model
    assert private_optimizer == mock_private_optimizer
    assert private_data_loader == mock_private_data_loader
    assert privacy_engine == mock_privacy_engine
    assert (
        private_loss_fn == mock_private_loss_fn
    )  # Should not be None for fast clipping

    # Verify make_private was called with correct parameters including kwargs
    mock_privacy_engine.make_private.assert_called_once_with(
        module=simple_model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=dp_config_with_fast_clipping.noise_multiplier,
        max_grad_norm=dp_config_with_fast_clipping.max_grad_norm,
        clipping="fast",
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.utils.trainer_utils.differential_privacy_utils.PrivacyEngine")
def test_setup_privacy_engine_with_additional_kwargs(
    mock_privacy_engine_class, simple_model, optimizer, data_loader
):
    """Test setup_privacy_engine with additional kwargs."""
    # Create a DP config with additional kwargs
    dp_config = DifferentialPrivacyBase(
        enable=True,
        noise_multiplier=1.5,
        max_grad_norm=2.0,
        delta=1e-6,
        kwargs={"poisson_sampling": True, "batch_first": False},
    )

    # Create a mock privacy engine instance
    mock_privacy_engine = MagicMock()
    mock_privacy_engine_class.return_value = mock_privacy_engine

    # Mock the make_private method to return 3 items (standard clipping)
    mock_privacy_engine.make_private.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    # Call the function
    setup_privacy_engine(simple_model, optimizer, data_loader, dp_config)

    # Verify make_private was called with correct parameters including additional kwargs
    mock_privacy_engine.make_private.assert_called_once_with(
        module=simple_model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.5,
        max_grad_norm=2.0,
        poisson_sampling=True,
        batch_first=False,
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.utils.trainer_utils.differential_privacy_utils.PrivacyEngine")
def test_setup_privacy_engine_fast_clipping_wrong_return_count(
    mock_privacy_engine_class,
    simple_model,
    optimizer,
    data_loader,
    dp_config_with_fast_clipping,
):
    """Test setup_privacy_engine with fast clipping returning wrong number of items."""
    # Create a mock privacy engine instance
    mock_privacy_engine = MagicMock()
    mock_privacy_engine_class.return_value = mock_privacy_engine

    # Mock the make_private method to return wrong number of items for fast clipping
    mock_privacy_engine.make_private.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),  # Only 3 items instead of expected 4 for fast clipping
    )

    # Call the function and expect ValueError
    with pytest.raises(
        ValueError,
        match="Opacus with 'fast' clipping did not return the expected 4 values.",
    ):
        setup_privacy_engine(
            simple_model, optimizer, data_loader, dp_config_with_fast_clipping
        )


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.utils.trainer_utils.differential_privacy_utils.PrivacyEngine")
def test_setup_privacy_engine_standard_clipping_wrong_return_count(
    mock_privacy_engine_class, simple_model, optimizer, data_loader, dp_config
):
    """Test setup_privacy_engine with standard clipping returning wrong number of items."""
    # Create a mock privacy engine instance
    mock_privacy_engine = MagicMock()
    mock_privacy_engine_class.return_value = mock_privacy_engine

    # Mock the make_private method to return wrong number of items for standard clipping
    mock_privacy_engine.make_private.return_value = (
        MagicMock(),
        MagicMock(),  # Only 2 items instead of expected 3 for standard clipping
    )

    # Call the function and expect ValueError
    with pytest.raises(
        ValueError, match="Opacus did not return the expected 3 values."
    ):
        setup_privacy_engine(simple_model, optimizer, data_loader, dp_config)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.utils.trainer_utils.differential_privacy_utils.PrivacyEngine")
def test_setup_privacy_engine_no_kwargs(
    mock_privacy_engine_class, simple_model, optimizer, data_loader
):
    """Test setup_privacy_engine when kwargs is None."""
    # Create a DP config with None kwargs
    dp_config = DifferentialPrivacyBase(
        enable=True,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        delta=1e-5,
        kwargs=None,
    )

    # Create a mock privacy engine instance
    mock_privacy_engine = MagicMock()
    mock_privacy_engine_class.return_value = mock_privacy_engine

    # Mock the make_private method to return 3 items (standard clipping)
    mock_privacy_engine.make_private.return_value = (
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )

    # Call the function
    setup_privacy_engine(simple_model, optimizer, data_loader, dp_config)

    # Verify make_private was called with correct parameters (no additional kwargs)
    mock_privacy_engine.make_private.assert_called_once_with(
        module=simple_model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=dp_config.noise_multiplier,
        max_grad_norm=dp_config.max_grad_norm,
    )


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
@patch("advsecurenet.utils.trainer_utils.differential_privacy_utils.PrivacyEngine")
def test_setup_privacy_engine_integration_test(
    mock_privacy_engine_class, simple_model, optimizer, data_loader, dp_config
):
    """Integration test to ensure all components work together correctly."""
    # Create a mock privacy engine instance
    mock_privacy_engine = MagicMock()
    mock_privacy_engine_class.return_value = mock_privacy_engine

    # Create mock returns that maintain some properties
    mock_private_model = MagicMock()
    mock_private_model.parameters.return_value = simple_model.parameters()
    mock_private_optimizer = MagicMock()
    mock_private_data_loader = MagicMock()
    mock_private_data_loader.__len__ = lambda: len(data_loader)

    mock_privacy_engine.make_private.return_value = (
        mock_private_model,
        mock_private_optimizer,
        mock_private_data_loader,
    )

    # Call the function
    result = setup_privacy_engine(simple_model, optimizer, data_loader, dp_config)

    # Comprehensive verification
    (
        private_model,
        private_optimizer,
        private_data_loader,
        privacy_engine,
        private_loss_fn,
    ) = result

    # Verify types and structure
    assert private_model == mock_private_model
    assert private_optimizer == mock_private_optimizer
    assert private_data_loader == mock_private_data_loader
    assert privacy_engine == mock_privacy_engine
    assert private_loss_fn is None

    # Verify the privacy engine was properly instantiated
    mock_privacy_engine_class.assert_called_once()

    # Verify make_private was called with all required arguments
    call_args = mock_privacy_engine.make_private.call_args
    assert call_args[1]["module"] == simple_model
    assert call_args[1]["optimizer"] == optimizer
    assert call_args[1]["data_loader"] == data_loader
    assert call_args[1]["noise_multiplier"] == dp_config.noise_multiplier
    assert call_args[1]["max_grad_norm"] == dp_config.max_grad_norm
