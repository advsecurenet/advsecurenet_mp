import pytest

from advsecurenet.utils.huggingface_utils import huggingface_general_utils
from advsecurenet.utils.huggingface_utils.huggingface_dataset_utils import (
    check_hub_for_dataset_id,
    verify_hf_dataset_identifier_exists,
)
from advsecurenet.utils.huggingface_utils.huggingface_model_utils import (
    check_hub_for_model_id,
    verify_hf_model_identifier_exists,
)

# --- huggingface_general_utils.py ---


@pytest.mark.parametrize(
    "is_hf_url,hf_url_expected_existence",
    [
        ("https://huggingface.co/microsoft/resnet-18", True),
        ("https://huggingface.co/datasets/uoft-cs/cifar10", True),
        ("https://hf.co/uoft-cs/cifar10", True),
        ("https://example.com/not-hf", False),
        ("", False),
        (None, False),
    ],
)
def test_is_huggingface_url(is_hf_url, hf_url_expected_existence):
    assert (
        huggingface_general_utils.is_huggingface_url(is_hf_url)
        == hf_url_expected_existence
    )


@pytest.mark.parametrize(
    "url_without_scheme,expected_result",
    [
        ("huggingface.co/microsoft/resnet-18", True),
        ("hf.co/microsoft/resnet-18", True),
        ("www.huggingface.co/datasets/uoft-cs/cifar10", True),
        ("example.com/not-hf", False),
        ("huggingface.co", False),  # Not enough path segments
        ("hf.co/single-segment", False),  # Not enough path segments
    ],
)
def test_is_huggingface_url_without_scheme(url_without_scheme, expected_result):
    """Test URLs without scheme to cover the url = 'https://' + url line"""
    assert (
        huggingface_general_utils.is_huggingface_url(url_without_scheme)
        == expected_result
    )


@pytest.mark.parametrize(
    "hf_id,hf_id_expected_existence",
    [
        ("microsoft/resnet-18", True),
        ("uoft-cs/cifar10", True),
        ("invalid-id", False),
        ("", False),
        (None, False),
    ],
)
def test_is_huggingface_id(hf_id, hf_id_expected_existence):
    assert (
        huggingface_general_utils.is_huggingface_id(hf_id) == hf_id_expected_existence
    )


@pytest.mark.parametrize(
    "to_extract_hf_url,extracted_hf_url",
    [
        ("https://huggingface.co/microsoft/resnet-18", "microsoft/resnet-18"),
        ("https://huggingface.co/datasets/uoft-cs/cifar10", "uoft-cs/cifar10"),
        ("https://hf.co/uoft-cs/cifar10", "uoft-cs/cifar10"),
        ("https://example.com/not-hf", None),
        ("", None),
        (None, None),
    ],
)
def test_extract_id_from_url(to_extract_hf_url, extracted_hf_url):
    assert (
        huggingface_general_utils.extract_id_from_url(to_extract_hf_url)
        == extracted_hf_url
    )


@pytest.mark.parametrize(
    "url_without_scheme,expected_extracted_id",
    [
        ("huggingface.co/microsoft/resnet-18", "microsoft/resnet-18"),
        ("hf.co/microsoft/resnet-18", "microsoft/resnet-18"),
        ("www.huggingface.co/datasets/uoft-cs/cifar10", "uoft-cs/cifar10"),
        ("huggingface.co/datasets/test/dataset", "test/dataset"),
        ("example.com/not-hf", None),  # Not a valid HF URL
        ("huggingface.co", None),  # Not enough path segments
    ],
)
def test_extract_id_from_url_without_scheme(url_without_scheme, expected_extracted_id):
    """Test URL extraction without scheme to cover the url = 'https://' + url line"""
    assert (
        huggingface_general_utils.extract_id_from_url(url_without_scheme)
        == expected_extracted_id
    )


@pytest.mark.parametrize(
    "to_process_hf_identifier,processed_hf_identifier",
    [
        ("https://huggingface.co/microsoft/resnet-18", "microsoft/resnet-18"),
        ("https://huggingface.co/datasets/uoft-cs/cifar10", "uoft-cs/cifar10"),
        ("uoft-cs/cifar10", "uoft-cs/cifar10"),
        ("microsoft/resnet-18", "microsoft/resnet-18"),
        ("invalid-id", None),
        ("", None),
        (None, None),
    ],
)
def test_process_hf_identifier(to_process_hf_identifier, processed_hf_identifier):
    assert (
        huggingface_general_utils.process_hf_identifier(to_process_hf_identifier)
        == processed_hf_identifier
    )


# --- huggingface_dataset_hub_utils.py ---


@pytest.mark.integration
@pytest.mark.parametrize(
    "to_exist_hf_dataset_id,dataset_id_expected_existence",
    [
        ("uoft-cs/cifar10", True),
        ("nonexistent-user/nonexistent-dataset", False),
    ],
)
def test_check_hub_for_dataset_id(
    to_exist_hf_dataset_id, dataset_id_expected_existence
):
    assert (
        check_hub_for_dataset_id(to_exist_hf_dataset_id)
        == dataset_id_expected_existence
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "to_exist_hf_dataset_identifier,dataset_identifier_expected_existence",
    [
        ("uoft-cs/cifar10", True),
        ("https://huggingface.co/datasets/uoft-cs/cifar10", True),
        ("https://hf.co/datasets/uoft-cs/cifar10", True),
        ("nonexistent-user/nonexistent-dataset", False),
        ("https://huggingface.co/datasets/nonexistent-user/nonexistent-dataset", False),
        ("https://hf.co/datasets/nonexistent-user/nonexistent-dataset", False),
        ("invalid-id", False),
        ("", False),
        (None, False),
    ],
)
def test_verify_hf_dataset_identifier_exists(
    to_exist_hf_dataset_identifier, dataset_identifier_expected_existence
):
    assert (
        verify_hf_dataset_identifier_exists(to_exist_hf_dataset_identifier)
        == dataset_identifier_expected_existence
    )


@pytest.mark.parametrize(
    "test_identifier,expected_warning_message",
    [
        ("uoft-cs/cifar10", "Could not verify Hugging Face identifier 'uoft-cs/cifar10' due to Hub check error: Test exception"),
        ("https://huggingface.co/datasets/test/dataset", "Could not verify Hugging Face identifier 'https://huggingface.co/datasets/test/dataset' due to Hub check error: Test exception"),
    ],
)
def test_verify_hf_dataset_identifier_exists_exception_handling(test_identifier, expected_warning_message):
    """Test exception handling in verify_hf_dataset_identifier_exists"""
    from unittest.mock import patch
    import warnings
    
    with patch('advsecurenet.utils.huggingface_utils.huggingface_dataset_utils.huggingface_dataset_hub_utils.check_hub_for_dataset_id') as mock_check:
        mock_check.side_effect = Exception("Test exception")
        
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")
            result = verify_hf_dataset_identifier_exists(test_identifier)
            
            # Should return False when exception occurs
            assert result is False
            
            # Should have issued exactly one warning
            assert len(warning_list) == 1
            
            # Check the warning message
            assert str(warning_list[0].message) == expected_warning_message


# --- huggingface_model_hub_utils.py ---


@pytest.mark.integration
@pytest.mark.parametrize(
    "to_check_model_id,model_id_expected_existence",
    [
        ("microsoft/resnet-18", True),
        ("bert-base-uncased", True),
        ("nonexistent-user/nonexistent-model", False),
    ],
)
def test_check_hub_for_model_id(to_check_model_id, model_id_expected_existence):
    assert check_hub_for_model_id(to_check_model_id) == model_id_expected_existence


@pytest.mark.integration
@pytest.mark.parametrize(
    "to_verify_hf_model_identifier,hf_model_identifier_expected_existence",
    [
        ("microsoft/resnet-18", True),
        ("https://huggingface.co/microsoft/resnet-18", True),
        ("google-bert/bert-base-uncased", True),
        ("https://hf.co/google-bert/bert-base-uncased", True),
        ("nonexistent-user/nonexistent-model", False),
        ("https://huggingface.co/nonexistent-user/nonexistent-model", False),
        ("invalid-id", False),
        ("", False),
        (None, False),
    ],
)
def test_verify_hf_model_identifier_exists(
    to_verify_hf_model_identifier, hf_model_identifier_expected_existence
):
    assert (
        verify_hf_model_identifier_exists(to_verify_hf_model_identifier)
        == hf_model_identifier_expected_existence
    )
