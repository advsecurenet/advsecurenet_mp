import pytest
import pandas as pd
import tempfile
from unittest.mock import MagicMock, patch, mock_open
from advsecurenet.llm.GCG.src.conversation.template_utils import get_workers, get_goals_and_targets


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "</s>"
    tokenizer.unk_token = "<unk>"
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.unk_token_id = 0
    tokenizer.padding_side = 'right'
    return tokenizer


@pytest.fixture
def mock_conv_template():
    template = MagicMock()
    template.name = "test-template"
    template.roles = ("User", "Assistant")
    template.sep = "\n"
    template.sep2 = None
    return template


@pytest.fixture
def basic_params():
    params = MagicMock()
    params.tokenizer_paths = ["test/tokenizer"]
    params.tokenizer_kwargs = [{"use_fast": False}]
    params.conversation_templates = ["test-template"]
    params.model_paths = ["test/model"]
    params.model_kwargs = [{"low_cpu_mem_usage": True}]
    params.devices = ["cpu"]
    return params


class TestGetWorkers:
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.AutoTokenizer.from_pretrained')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.get_conversation_template')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.ModelWorker')
    def test_get_workers_basic(self, mock_worker_class, mock_get_template, mock_tokenizer_class, 
                             basic_params, mock_tokenizer, mock_conv_template):
        mock_tokenizer_class.return_value = mock_tokenizer
        mock_get_template.return_value = mock_conv_template
        mock_worker = MagicMock()
        mock_worker.start = MagicMock()
        mock_worker_class.return_value = mock_worker
        
        workers, test_workers = get_workers(basic_params)
        
        # Verify tokenizer loading
        mock_tokenizer_class.assert_called_once_with(
            "test/tokenizer",
            trust_remote_code=True,
            use_fast=False
        )
        
        # Verify template loading
        mock_get_template.assert_called_once_with("test-template")
        
        # Verify worker creation
        mock_worker_class.assert_called_once_with(
            "test/model",
            {"low_cpu_mem_usage": True},
            mock_tokenizer,
            mock_conv_template,
            "cpu"
        )
        
        # Verify worker start
        mock_worker.start.assert_called_once()
        
        assert len(workers) == 1
        assert len(test_workers) == 0

    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.AutoTokenizer.from_pretrained')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.get_conversation_template')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.ModelWorker')
    def test_get_workers_eval_mode(self, mock_worker_class, mock_get_template, mock_tokenizer_class,
                                 basic_params, mock_tokenizer, mock_conv_template):
        mock_tokenizer_class.return_value = mock_tokenizer
        mock_get_template.return_value = mock_conv_template
        mock_worker = MagicMock()
        mock_worker_class.return_value = mock_worker
        
        workers, test_workers = get_workers(basic_params, eval=True)
        
        # In eval mode, workers should not be started
        mock_worker.start.assert_not_called()

    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.AutoTokenizer.from_pretrained')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.get_conversation_template')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.ModelWorker')
    def test_get_workers_multiple_workers(self, mock_worker_class, mock_get_template, mock_tokenizer_class,
                                        mock_tokenizer, mock_conv_template):
        params = MagicMock()
        params.tokenizer_paths = ["tokenizer1", "tokenizer2", "tokenizer3"]
        params.tokenizer_kwargs = [{}, {}, {}]
        params.conversation_templates = ["template1", "template2", "template3"]
        params.model_paths = ["model1", "model2", "model3"]
        params.model_kwargs = [{}, {}, {}]
        params.devices = ["cpu", "cuda:0", "cuda:1"]
        params.num_train_models = 2
        
        mock_tokenizer_class.return_value = mock_tokenizer
        mock_get_template.return_value = mock_conv_template
        mock_worker = MagicMock()
        mock_worker_class.return_value = mock_worker
        
        workers, test_workers = get_workers(params)
        
        # Verify correct number of workers
        assert len(workers) == 2  # num_train_models
        assert len(test_workers) == 1  # remaining workers
        
        # Verify all workers were created
        assert mock_worker_class.call_count == 3

    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.AutoTokenizer.from_pretrained')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.get_conversation_template')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.ModelWorker')
    def test_tokenizer_special_configurations(self, mock_worker_class, mock_get_template, mock_tokenizer_class,
                                             mock_conv_template):
        # Test special tokenizer configurations
        test_cases = [
            ("oasst-sft-6-llama-30b", lambda t: (setattr(t, 'bos_token_id', 1), setattr(t, 'unk_token_id', 0))),
            ("guanaco", lambda t: (setattr(t, 'eos_token_id', 2), setattr(t, 'unk_token_id', 0))),
            ("llama-2", lambda t: (setattr(t, 'pad_token', t.unk_token), setattr(t, 'padding_side', 'left'))),
            ("falcon", lambda t: setattr(t, 'padding_side', 'left'))
        ]
        
        for model_name, config_fn in test_cases:
            params = MagicMock()
            params.tokenizer_paths = [model_name]
            params.tokenizer_kwargs = [{}]
            params.conversation_templates = ["test"]
            params.model_paths = ["test"]
            params.model_kwargs = [{}]
            params.devices = ["cpu"]
            
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer.unk_token = "<unk>"
            mock_tokenizer_class.return_value = mock_tokenizer
            mock_get_template.return_value = mock_conv_template
            mock_worker_class.return_value = MagicMock()
            
            get_workers(params)
            
            # Verify tokenizer was configured
            assert mock_tokenizer_class.called

    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.AutoTokenizer.from_pretrained')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.get_conversation_template')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.ModelWorker')
    def test_pad_token_fallback(self, mock_worker_class, mock_get_template, mock_tokenizer_class,
                               basic_params, mock_conv_template):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None  # No pad token
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.return_value = mock_tokenizer
        mock_get_template.return_value = mock_conv_template
        mock_worker_class.return_value = MagicMock()
        
        get_workers(basic_params)
        
        # Should set pad_token to eos_token
        assert mock_tokenizer.pad_token == mock_tokenizer.eos_token

    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.AutoTokenizer.from_pretrained')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.get_conversation_template')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.ModelWorker')
    def test_conversation_template_modifications(self, mock_worker_class, mock_get_template, mock_tokenizer_class,
                                               basic_params, mock_tokenizer):
        # Test zero_shot template modification
        zero_shot_template = MagicMock()
        zero_shot_template.name = 'zero_shot'
        zero_shot_template.roles = ['User', 'Assistant']
        zero_shot_template.sep = ' '
        
        mock_tokenizer_class.return_value = mock_tokenizer
        mock_get_template.return_value = zero_shot_template
        mock_worker_class.return_value = MagicMock()
        
        get_workers(basic_params)
        
        # Verify zero_shot modifications
        assert zero_shot_template.roles == ('### User', '### Assistant')
        assert zero_shot_template.sep == '\n'

    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.AutoTokenizer.from_pretrained')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.get_conversation_template')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.ModelWorker')
    def test_llama2_template_modification(self, mock_worker_class, mock_get_template, mock_tokenizer_class,
                                        basic_params, mock_tokenizer):
        # Test llama-2 template modification
        llama2_template = MagicMock()
        llama2_template.name = 'llama-2'
        llama2_template.sep2 = '  whitespace  '
        
        mock_tokenizer_class.return_value = mock_tokenizer
        mock_get_template.return_value = llama2_template
        mock_worker_class.return_value = MagicMock()
        
        get_workers(basic_params)
        
        # Verify llama-2 modifications
        assert llama2_template.sep2 == 'whitespace'  # Should be stripped

    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.AutoTokenizer.from_pretrained')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.get_conversation_template')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.ModelWorker')
    def test_num_train_models_default(self, mock_worker_class, mock_get_template, mock_tokenizer_class,
                                    mock_tokenizer, mock_conv_template):
        params = MagicMock()
        params.tokenizer_paths = ["t1", "t2", "t3"]
        params.tokenizer_kwargs = [{}, {}, {}]
        params.conversation_templates = ["c1", "c2", "c3"]
        params.model_paths = ["m1", "m2", "m3"]
        params.model_kwargs = [{}, {}, {}]
        params.devices = ["cpu", "cpu", "cpu"]
        # No num_train_models attribute
        del params.num_train_models
        
        mock_tokenizer_class.return_value = mock_tokenizer
        mock_get_template.return_value = mock_conv_template
        mock_worker_class.return_value = MagicMock()
        
        workers, test_workers = get_workers(params)
        
        # Should default to all workers as train workers
        assert len(workers) == 3
        assert len(test_workers) == 0


class TestGetGoalsAndTargets:
    def test_get_goals_targets_from_params(self):
        params = MagicMock()
        params.goals = ["goal1", "goal2"]
        params.targets = ["target1", "target2"]
        params.test_goals = ["test_goal"]
        params.test_targets = ["test_target"]
        params.train_data = None
        
        goals, targets, test_goals, test_targets = get_goals_and_targets(params)
        
        assert goals == ["goal1", "goal2"]
        assert targets == ["target1", "target2"]
        assert test_goals == ["test_goal"]
        assert test_targets == ["test_target"]

    def test_get_goals_targets_default_empty(self):
        params = MagicMock()
        # Remove all attributes to test defaults
        del params.goals
        del params.targets
        del params.test_goals
        del params.test_targets
        del params.data_offset
        params.train_data = None
        
        goals, targets, test_goals, test_targets = get_goals_and_targets(params)
        
        assert goals == []
        assert targets == []
        assert test_goals == []
        assert test_targets == []

    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.pd.read_csv')
    def test_get_goals_targets_from_csv_with_goals(self, mock_read_csv):
        params = MagicMock()
        params.train_data = "train.csv"
        params.test_data = ""
        params.n_train_data = 2
        params.n_test_data = 0
        params.data_offset = 0
        
        # Mock CSV with both goal and target columns
        mock_df = pd.DataFrame({
            'goal': ['goal1', 'goal2', 'goal3'],
            'target': ['target1', 'target2', 'target3']
        })
        mock_read_csv.return_value = mock_df
        
        goals, targets, test_goals, test_targets = get_goals_and_targets(params)
        
        assert goals == ['goal1', 'goal2']
        assert targets == ['target1', 'target2']
        assert test_goals == []
        assert test_targets == []

    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.pd.read_csv')
    def test_get_goals_targets_from_csv_no_goals(self, mock_read_csv):
        params = MagicMock()
        params.train_data = "train.csv"
        params.test_data = ""
        params.n_train_data = 2
        params.n_test_data = 0
        params.data_offset = 0
        
        # Mock CSV with only target column
        mock_df = pd.DataFrame({
            'target': ['target1', 'target2', 'target3']
        })
        mock_read_csv.return_value = mock_df
        
        goals, targets, test_goals, test_targets = get_goals_and_targets(params)
        
        assert goals == ["", ""]  # Empty strings for missing goals
        assert targets == ['target1', 'target2']

    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.pd.read_csv')
    def test_get_goals_targets_with_offset(self, mock_read_csv):
        params = MagicMock()
        params.train_data = "train.csv"
        params.test_data = ""
        params.n_train_data = 2
        params.n_test_data = 0
        params.data_offset = 1  # Skip first row
        
        mock_df = pd.DataFrame({
            'goal': ['goal0', 'goal1', 'goal2', 'goal3'],
            'target': ['target0', 'target1', 'target2', 'target3']
        })
        mock_read_csv.return_value = mock_df
        
        goals, targets, test_goals, test_targets = get_goals_and_targets(params)
        
        assert goals == ['goal1', 'goal2']  # Starting from offset 1
        assert targets == ['target1', 'target2']

    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.pd.read_csv')
    def test_get_goals_targets_with_separate_test_data(self, mock_read_csv):
        params = MagicMock()
        params.train_data = "train.csv"
        params.test_data = "test.csv"
        params.n_train_data = 2
        params.n_test_data = 1
        params.data_offset = 0
        
        train_df = pd.DataFrame({
            'goal': ['train_goal1', 'train_goal2'],
            'target': ['train_target1', 'train_target2']
        })
        test_df = pd.DataFrame({
            'goal': ['test_goal1'],
            'target': ['test_target1']
        })
        
        mock_read_csv.side_effect = [train_df, test_df]
        
        goals, targets, test_goals, test_targets = get_goals_and_targets(params)
        
        assert goals == ['train_goal1', 'train_goal2']
        assert targets == ['train_target1', 'train_target2']
        assert test_goals == ['test_goal1']
        assert test_targets == ['test_target1']

    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.pd.read_csv')
    def test_get_goals_targets_test_from_train_data(self, mock_read_csv):
        params = MagicMock()
        params.train_data = "train.csv"
        params.test_data = ""  # No separate test data
        params.n_train_data = 2
        params.n_test_data = 1  # Take test from train data
        params.data_offset = 0
        
        mock_df = pd.DataFrame({
            'goal': ['goal1', 'goal2', 'goal3'],
            'target': ['target1', 'target2', 'target3']
        })
        mock_read_csv.return_value = mock_df
        
        goals, targets, test_goals, test_targets = get_goals_and_targets(params)
        
        assert goals == ['goal1', 'goal2']  # First n_train_data
        assert targets == ['target1', 'target2']
        assert test_goals == ['goal3']  # Next n_test_data
        assert test_targets == ['target3']

    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.pd.read_csv')
    def test_assertion_error_mismatched_lengths(self, mock_read_csv):
        params = MagicMock()
        params.train_data = "train.csv"
        params.test_data = ""
        params.n_train_data = 2
        params.n_test_data = 0
        params.data_offset = 0
        
        # Create mismatched data
        mock_df = pd.DataFrame({
            'goal': ['goal1'],  # Only one goal
            'target': ['target1', 'target2']  # But two targets
        })
        mock_read_csv.return_value = mock_df
        
        with pytest.raises(AssertionError):
            get_goals_and_targets(params)

    def test_getattr_defaults(self):
        # Test that getattr provides proper defaults
        params = MagicMock()
        params.train_data = None
        # Remove all optional attributes
        for attr in ['goals', 'targets', 'test_goals', 'test_targets', 'data_offset']:
            if hasattr(params, attr):
                delattr(params, attr)
        
        goals, targets, test_goals, test_targets = get_goals_and_targets(params)
        
        # Should use defaults without error
        assert isinstance(goals, list)
        assert isinstance(targets, list)
        assert isinstance(test_goals, list)
        assert isinstance(test_targets, list)


class TestTemplateUtilsIntegration:
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.AutoTokenizer.from_pretrained')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.get_conversation_template')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.ModelWorker')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.pd.read_csv')
    def test_full_workflow_integration(self, mock_read_csv, mock_worker_class, mock_get_template, 
                                     mock_tokenizer_class):
        # Test complete workflow with both functions
        
        # Setup params
        params = MagicMock()
        params.tokenizer_paths = ["test/tokenizer"]
        params.tokenizer_kwargs = [{}]
        params.conversation_templates = ["test"]
        params.model_paths = ["test/model"]
        params.model_kwargs = [{}]
        params.devices = ["cpu"]
        params.num_train_models = 1
        params.train_data = "data.csv"
        params.test_data = ""
        params.n_train_data = 2
        params.n_test_data = 0
        params.data_offset = 0
        
        # Mock components
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "</s>"
        mock_tokenizer_class.return_value = mock_tokenizer
        
        mock_template = MagicMock()
        mock_template.name = "test"
        mock_get_template.return_value = mock_template
        
        mock_worker = MagicMock()
        mock_worker_class.return_value = mock_worker
        
        mock_df = pd.DataFrame({
            'goal': ['goal1', 'goal2'],
            'target': ['target1', 'target2']
        })
        mock_read_csv.return_value = mock_df
        
        # Run both functions
        workers, test_workers = get_workers(params)
        goals, targets, test_goals, test_targets = get_goals_and_targets(params)
        
        # Verify integration
        assert len(workers) == 1
        assert len(test_workers) == 0
        assert goals == ['goal1', 'goal2']
        assert targets == ['target1', 'target2']


class TestTemplateUtilsEdgeCases:
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.AutoTokenizer.from_pretrained')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.get_conversation_template')
    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.ModelWorker')
    def test_empty_worker_lists(self, mock_worker_class, mock_get_template, mock_tokenizer_class):
        params = MagicMock()
        params.tokenizer_paths = []
        params.tokenizer_kwargs = []
        params.conversation_templates = []
        params.model_paths = []
        params.model_kwargs = []
        params.devices = []
        
        workers, test_workers = get_workers(params)
        
        assert len(workers) == 0
        assert len(test_workers) == 0

    def test_csv_file_not_found(self):
        params = MagicMock()
        params.train_data = "nonexistent.csv"
        params.n_train_data = 1
        params.data_offset = 0
        
        with patch('advsecurenet.llm.GCG.src.conversation.template_utils.pd.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = FileNotFoundError("File not found")
            
            with pytest.raises(FileNotFoundError):
                get_goals_and_targets(params)

    @patch('advsecurenet.llm.GCG.src.conversation.template_utils.pd.read_csv')
    def test_empty_csv_file(self, mock_read_csv):
        params = MagicMock()
        params.train_data = "empty.csv"
        params.test_data = ""
        params.n_train_data = 1
        params.n_test_data = 0
        params.data_offset = 0
        
        # Empty DataFrame
        mock_read_csv.return_value = pd.DataFrame()
        
        with pytest.raises(KeyError):  # Should fail when trying to access 'target' column
            get_goals_and_targets(params)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])