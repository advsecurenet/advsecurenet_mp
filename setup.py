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
        "scikit-learn",
        "einops",
        "filetype",
        "requests",
        "pydantic>=2,<3",
        "transformers==4.48.3",
        "fschat==0.2.36",
        "datasets",
        "mean_average_precision",
        "yolov5",
        "huggingface-hub==0.24.1",  # Ensure compatibility with YOLOv5
        "pycocotools",
        "opacus",
        "numpy>=1.21.0,<1.25.0",
        "peft",
        "bitsandbytes"
    ],
    entry_points={
        "console_scripts": [
            "advsecurenet=cli.cli:main",
        ],
    },
)
