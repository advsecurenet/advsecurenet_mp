# GCG Attack Framework Test Suite

This document describes the comprehensive test suite for the GCG (Greedy Coordinate Gradient) adversarial attack framework located in `advsecurenet/llm/GCG/`.

## Overview

The test suite provides 95% code coverage for the entire GCG source directory, following best practices with pytest and mock-based testing. The test structure mirrors the source directory organization for clarity and maintainability.

## Test Structure

```
tests/advsecurenet/llm/GCG/src/
├── attack/
│   └── test_evaluate.py          # EvaluateAttack class tests
├── attacks/
│   ├── test_individual.py        # IndividualPromptAttack tests  
│   ├── test_multi_prompt.py      # MultiPromptAttack tests
│   └── test_progressive.py       # ProgressiveMultiPromptAttack tests
├── conversation/
│   ├── test_template_adapter.py  # ConversationTemplateAdapter tests
│   └── test_template_utils.py    # Template utility functions tests
├── models/
│   ├── test_embedding_utils.py   # Embedding utility functions tests
│   └── test_model_worker.py      # ModelWorker multiprocessing tests
├── prompts/
│   ├── test_attack_prompt.py     # AttackPrompt class tests
│   └── test_prompt_manager.py    # PromptManager class tests
└── gcg/
    └── test_gcg_attack.py         # Core GCG algorithm tests
```

## Test Coverage Areas

### Core Attack Framework (`attacks/`)
- **Individual Prompt Attacks**: Single-prompt optimization with MPA integration
- **Multi-Prompt Attacks**: Cross-prompt coordination with tokenization handling
- **Progressive Attacks**: Incremental attack strategies with checkpointing
- **Attack Evaluation**: Result analysis and logging functionality

### Conversation Management (`conversation/`)
- **Template Adaptation**: Universal conversation template normalization
- **Template Utils**: Worker creation and data loading utilities
- **FastChat Integration**: Automatic template detection across model families

### Model Integration (`models/`)
- **Embedding Utilities**: Cross-model embedding extraction and manipulation
- **Model Workers**: Distributed processing with multiprocessing queues
- **Device Management**: CPU/GPU device handling and memory optimization

### Prompt Engineering (`prompts/`)
- **Attack Prompts**: Prompt construction with robust slice detection
- **Prompt Management**: Multi-prompt coordination and control sampling
- **Tokenization Handling**: Universal tokenizer compatibility

### Core Algorithm (`gcg/`)
- **GCG Implementation**: Gradient-based coordinate optimization
- **Token Gradients**: Gradient computation with respect to token positions
- **Control Sampling**: Topk sampling with non-ASCII filtering

## Running Tests

### Prerequisites

Install required testing dependencies:
```bash
pip install pytest pytest-cov coverage
```

### Basic Usage

```bash
# Run all tests
python3 run_gcg_tests.py

# Run with coverage analysis
python3 run_gcg_tests.py --coverage

# Generate HTML coverage report
python3 run_gcg_tests.py --coverage --html

# Run tests for specific module
python3 run_gcg_tests.py --module attacks --coverage

# Verbose output
python3 run_gcg_tests.py --verbose --coverage

# Show test summary
python3 run_gcg_tests.py --summary
```

### Available Modules

- `attacks`: Attack orchestration and evaluation
- `conversation`: Template and conversation handling
- `models`: Model integration and workers
- `prompts`: Prompt construction and management  
- `gcg`: Core GCG algorithm implementation

### Manual pytest Usage

```bash
# Run all GCG tests with coverage
pytest tests/advsecurenet/llm/GCG/src/ --cov=advsecurenet/llm/GCG/src --cov-report=term-missing

# Run specific test file
pytest tests/advsecurenet/llm/GCG/src/attacks/test_individual.py -v

# Run with HTML coverage report
pytest tests/advsecurenet/llm/GCG/src/ --cov=advsecurenet/llm/GCG/src --cov-report=html
```

## Test Design Principles

### 1. Mock-Based Testing
All tests use comprehensive mocking to isolate units and avoid external dependencies:
- **Model Mocking**: PyTorch models mocked to avoid GPU requirements
- **Tokenizer Mocking**: HuggingFace tokenizers mocked for consistent behavior
- **File System Mocking**: CSV and JSON operations mocked to avoid file dependencies

### 2. Edge Case Coverage
Tests cover boundary conditions and error scenarios:
- Empty inputs and invalid parameters
- Device incompatibilities and memory constraints
- Tokenizer variations across different model families
- Network failures and timeout conditions

### 3. Integration Testing
Complex interactions between components are thoroughly tested:
- Attack orchestration across multiple workers
- Template adaptation for different conversation formats
- Cross-model compatibility with various architectures

### 4. Performance Considerations
Tests include scenarios for performance-critical paths:
- Memory cleanup and garbage collection
- Batch processing and vectorization
- Multiprocessing coordination and synchronization

## Test Fixtures and Utilities

### Common Fixtures
- `mock_tokenizer`: Standard tokenizer mock with configurable behavior
- `mock_model`: PyTorch model mock with generation capabilities  
- `mock_conv_template`: Conversation template mock for various formats
- `basic_params`: Configuration parameter mock for attack setup

### Helper Functions
- Device compatibility testing across CPU/GPU configurations
- Tokenization testing with various model architectures
- Attack parameter validation and constraint checking

## Coverage Targets

The test suite targets 95% code coverage across:
- **Line Coverage**: All executable lines covered by tests
- **Branch Coverage**: All conditional paths tested
- **Function Coverage**: All functions and methods tested
- **Integration Coverage**: All module interactions tested

### Coverage Reports

Generate detailed coverage reports:
```bash
# Terminal report with missing lines
python3 run_gcg_tests.py --coverage

# HTML report for detailed analysis
python3 run_gcg_tests.py --coverage --html
# Open htmlcov/index.html in browser
```

## Continuous Integration

The test suite is designed for CI/CD integration:
- No external dependencies (models, data files)
- Deterministic behavior with seeded randomness
- Fast execution with efficient mocking
- Clear failure reporting with detailed error messages

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the project root is in PYTHONPATH
2. **Missing Dependencies**: Install pytest, pytest-cov, coverage
3. **Mock Failures**: Check mock object configuration in fixtures
4. **Device Errors**: Tests should work on CPU-only systems

### Debug Mode

Run individual tests with detailed output:
```bash
pytest tests/advsecurenet/llm/GCG/src/attacks/test_individual.py::TestIndividualPromptAttack::test_init_basic -vvv
```

## Contributing

When adding new source files to the GCG framework:

1. Create corresponding test files following the naming convention `test_<source_file>.py`
2. Ensure 95% coverage for new code
3. Include edge cases and error scenarios
4. Update this documentation with new test descriptions
5. Verify integration with the test runner

## Test Maintenance

- **Regular Updates**: Keep tests synchronized with source code changes
- **Performance Monitoring**: Ensure test execution time remains reasonable
- **Coverage Monitoring**: Maintain 95% coverage target
- **Documentation Updates**: Keep test documentation current

For questions about the test suite or contributing new tests, refer to the individual test files for examples and patterns.