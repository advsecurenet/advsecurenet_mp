import os

from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="advsecurenet",
    version="0.2.3",
    description="AdvSecureNet | Adversarial Secure Networks | Machine Learning Security",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Melih Catal",
    author_email="melihcatal@gmail.com",
    url="https://github.com/melihcatal/advsecurenet",
    packages=find_packages(),
    include_package_data=True,
    package_data={"": ["*.yml"]},
    python_requires=">=3.10",
    install_requires=[
        "click",
        "torch",
        "torchvision",
        "colored",
        "tqdm",
        "PyYAML",
        "opencv-python",
        "ruamel.yaml",
        "matplotlib",
        "scikit-image",
        "einops",
        "filetype",
        "requests",
        "fschat==0.2.20",
        "nltk==3.8.1",
        "numpy==1.26.0",
        "openai==0.28.1",
        "transformers==4.28.0",
        "sentencepiece==0.1.99",
        "protobuf==4.24.4",
        "accelerate==0.23.0",
        "ml_collections",

    ],
    entry_points={
        "console_scripts": [
            "advsecurenet=cli.cli:main",
        ],
    },
)
