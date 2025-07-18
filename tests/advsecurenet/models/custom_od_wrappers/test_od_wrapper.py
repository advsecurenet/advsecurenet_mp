import numpy as np
import pytest
from advsecurenet.models.CustomODWrappers.ODWrapper import ODWrapper

class DummyODWrapper(ODWrapper):
    def compute_object_vanishing_gradient(self, x, training=False):
        return x
    def compute_object_untargeted_gradient(self, x, detections=None):
        return x
    def compute_object_fabrication_gradient(self, x, detections=None):
        return x
    def compute_object_mislabeling_gradient(self, x, detections=None):
        return x

def test_odwrapper_constructor():
    model = object()
    conf_thresh = 0.5
    device_type = 'cpu'
    clip_values = (0, 1)
    input_shape = (3, 224, 224)
    wrapper = DummyODWrapper(model, conf_thresh, device_type, clip_values, input_shape)
    assert wrapper.model is model
    assert wrapper.conf_thresh == conf_thresh
    assert wrapper.device == device_type
    assert wrapper.clip_values == clip_values
    assert wrapper.input_shape == input_shape

def test_filter_boxes_all_below_thresh():
    wrapper = DummyODWrapper(None, 0.5, 'cpu', (0, 1), (3, 224, 224))
    predictions = {
        'boxes': [np.array([1,2,3,4]), np.array([5,6,7,8])],
        'scores': np.array([0.1, 0.2]),
        'labels': np.array([1, 2])
    }
    result = wrapper.filter_boxes(predictions, conf_thresh=0.5)
    assert result == {}

def test_filter_boxes_some_above_thresh():
    wrapper = DummyODWrapper(None, 0.5, 'cpu', (0, 1), (3, 224, 224))
    predictions = {
        'boxes': [np.array([1,2,3,4]), np.array([5,6,7,8])],
        'scores': np.array([0.6, 0.2]),
        'labels': np.array([1, 2])
    }
    result = wrapper.filter_boxes(predictions, conf_thresh=0.5)
    assert 'boxes' in result and 'scores' in result and 'labels' in result
    np.testing.assert_array_equal(result['boxes'], np.array([[1,2,3,4]]))
    np.testing.assert_array_equal(result['scores'], np.array([0.6]))
    np.testing.assert_array_equal(result['labels'], np.array([1]))

def test_filter_boxes_all_above_thresh():
    wrapper = DummyODWrapper(None, 0.5, 'cpu', (0, 1), (3, 224, 224))
    predictions = {
        'boxes': [np.array([1,2,3,4]), np.array([5,6,7,8])],
        'scores': np.array([0.6, 0.7]),
        'labels': np.array([1, 2])
    }
    result = wrapper.filter_boxes(predictions, conf_thresh=0.5)
    assert 'boxes' in result and 'scores' in result and 'labels' in result
    np.testing.assert_array_equal(result['boxes'], np.array([[1,2,3,4],[5,6,7,8]]))
    np.testing.assert_array_equal(result['scores'], np.array([0.6, 0.7]))
    np.testing.assert_array_equal(result['labels'], np.array([1, 2]))

def test_filter_boxes_empty_predictions():
    wrapper = DummyODWrapper(None, 0.5, 'cpu', (0, 1), (3, 224, 224))
    predictions = {'boxes': [], 'scores': np.array([]), 'labels': np.array([])}
    result = wrapper.filter_boxes(predictions, conf_thresh=0.5)
    assert result == {}

def test_filter_boxes_missing_keys():
    wrapper = DummyODWrapper(None, 0.5, 'cpu', (0, 1), (3, 224, 224))
    # Missing 'scores'
    predictions = {'boxes': [np.array([1,2,3,4])], 'labels': np.array([1])}
    with pytest.raises(KeyError):
        wrapper.filter_boxes(predictions, conf_thresh=0.5)
    # Missing 'boxes'
    predictions = {'scores': np.array([0.6]), 'labels': np.array([1])}
    with pytest.raises(KeyError):
        wrapper.filter_boxes(predictions, conf_thresh=0.5)
    # Missing 'labels'
    predictions = {'boxes': [np.array([1,2,3,4])], 'scores': np.array([0.6])}
    with pytest.raises(KeyError):
        wrapper.filter_boxes(predictions, conf_thresh=0.5)

def test_filter_boxes_mismatched_lengths():
    wrapper = DummyODWrapper(None, 0.5, 'cpu', (0, 1), (3, 224, 224))
    predictions = {
        'boxes': [np.array([1,2,3,4])],
        'scores': np.array([0.6, 0.7]),
        'labels': np.array([1])
    }
    with pytest.raises(IndexError):
        wrapper.filter_boxes(predictions, conf_thresh=0.5)

def test_filter_boxes_non_numpy_types():
    wrapper = DummyODWrapper(None, 0.5, 'cpu', (0, 1), (3, 224, 224))
    predictions = {
        'boxes': [[1,2,3,4], [5,6,7,8]],
        'scores': [0.6, 0.7],
        'labels': [1, 2]
    }
    # Should raise TypeError due to list indexing with [i]
    with pytest.raises(TypeError):
        wrapper.filter_boxes(predictions, conf_thresh=0.5)

def test_filter_boxes_score_equals_thresh():
    wrapper = DummyODWrapper(None, 0.5, 'cpu', (0, 1), (3, 224, 224))
    predictions = {
        'boxes': [np.array([1,2,3,4])],
        'scores': np.array([0.5]),
        'labels': np.array([1])
    }
    result = wrapper.filter_boxes(predictions, conf_thresh=0.5)
    assert 'boxes' in result and 'scores' in result and 'labels' in result
    np.testing.assert_array_equal(result['boxes'], np.array([[1,2,3,4]]))
    np.testing.assert_array_equal(result['scores'], np.array([0.5]))
    np.testing.assert_array_equal(result['labels'], np.array([1]))

def test_odwrapper_abstract_methods_raise():
    class OnlyAbstractODWrapper(ODWrapper):
        pass
    with pytest.raises(TypeError):
        OnlyAbstractODWrapper(None, 0.5, 'cpu', (0, 1), (3, 224, 224))

def test_odwrapper_abstract_methods_raise_not_implemented():
    # Instantiating a subclass without implementing abstract methods raises TypeError
    class PartiallyImplementedODWrapper(ODWrapper):
        pass
    with pytest.raises(TypeError):
        PartiallyImplementedODWrapper(None, 0.5, 'cpu', (0, 1), (3, 224, 224))
