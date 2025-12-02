#!/usr/bin/env python3
"""
Comprehensive test runner for GCG (Greedy Coordinate Gradient) attack framework.
This script runs all tests for the GCG source directory with coverage analysis.

Usage:
    python run_gcg_tests.py [options]
    
Options:
    --coverage    Run tests with coverage analysis
    --verbose     Verbose test output
    --html        Generate HTML coverage report
    --module      Run tests for specific module (attacks, conversation, models, prompts, gcg)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent
GCG_TEST_DIR = PROJECT_ROOT / "tests" / "advsecurenet" / "llm" / "GCG" / "src"
GCG_SRC_DIR = PROJECT_ROOT / "advsecurenet" / "llm" / "GCG" / "src"

# Test modules mapping
TEST_MODULES = {
    "attacks": [
        "test_individual.py",
        "test_multi_prompt.py", 
        "test_progressive.py",
        "test_evaluate.py"  # Note: this is in attack/ directory
    ],
    "conversation": [
        "test_template_adapter.py",
        "test_template_utils.py"
    ],
    "models": [
        "test_model_worker.py",
        "test_embedding_utils.py"
    ],
    "prompts": [
        "test_attack_prompt.py",
        "test_prompt_manager.py"
    ],
    "gcg": [
        "test_gcg_attack.py"
    ]
}

def check_dependencies():
    """Check if required test dependencies are installed."""
    required_packages = ['pytest', 'pytest-cov', 'coverage']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print(f"\nInstall with: pip install {' '.join(missing_packages)}")
        return False
    return True

def run_tests(module=None, coverage=False, verbose=False, html_report=False):
    """Run GCG tests with optional coverage analysis."""
    
    if not check_dependencies():
        return False
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if module:
        if module not in TEST_MODULES:
            print(f"Unknown module: {module}")
            print(f"Available modules: {', '.join(TEST_MODULES.keys())}")
            return False
        
        # Add specific module tests
        if module == "attacks":
            # Special case: evaluate test is in attack/ directory
            test_files = []
            for test_file in TEST_MODULES[module]:
                if test_file == "test_evaluate.py":
                    test_files.append(str(GCG_TEST_DIR / "attack" / test_file))
                else:
                    test_files.append(str(GCG_TEST_DIR / module / test_file))
            cmd.extend(test_files)
        else:
            cmd.append(str(GCG_TEST_DIR / module))
    else:
        # Run all tests
        cmd.append(str(GCG_TEST_DIR))
    
    if coverage:
        cmd.extend([
            "--cov=" + str(GCG_SRC_DIR),
            "--cov-report=term-missing"
        ])
        
        if html_report:
            cmd.extend([
                "--cov-report=html:htmlcov"
            ])
    
    if verbose:
        cmd.append("-v")
    
    # Add output formatting
    cmd.extend(["-ra", "--tb=short"])
    
    print("Running command:", " ".join(cmd))
    print("=" * 70)
    
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        return result.returncode == 0
    except subprocess.SubprocessError as e:
        print(f"Error running tests: {e}")
        return False

def print_test_summary():
    """Print a summary of available tests."""
    print("GCG Test Suite Summary")
    print("=" * 50)
    print(f"Test directory: {GCG_TEST_DIR}")
    print(f"Source directory: {GCG_SRC_DIR}")
    print()
    
    total_tests = 0
    for module, tests in TEST_MODULES.items():
        print(f"{module:>12}: {len(tests)} test files")
        total_tests += len(tests)
        for test in tests:
            print(f"              - {test}")
    
    print(f"\nTotal test files: {total_tests}")
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Run GCG attack framework tests with comprehensive coverage analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_gcg_tests.py --coverage --verbose
  python run_gcg_tests.py --module attacks --coverage
  python run_gcg_tests.py --html --coverage
        """
    )
    
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run tests with coverage analysis"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose test output"
    )
    parser.add_argument(
        "--html", 
        action="store_true", 
        help="Generate HTML coverage report (requires --coverage)"
    )
    parser.add_argument(
        "--module", 
        choices=list(TEST_MODULES.keys()),
        help="Run tests for specific module only"
    )
    parser.add_argument(
        "--summary", 
        action="store_true", 
        help="Show test summary and exit"
    )
    
    args = parser.parse_args()
    
    if args.summary:
        print_test_summary()
        return
    
    if args.html and not args.coverage:
        print("--html requires --coverage")
        sys.exit(1)
    
    print_test_summary()
    
    success = run_tests(
        module=args.module,
        coverage=args.coverage,
        verbose=args.verbose,
        html_report=args.html
    )
    
    if args.coverage and args.html:
        html_path = PROJECT_ROOT / "htmlcov" / "index.html"
        if html_path.exists():
            print(f"\nHTML coverage report generated: {html_path}")
    
    if not success:
        print("\nTests failed!")
        sys.exit(1)
    else:
        print("\nAll tests passed!")

if __name__ == "__main__":
    main()