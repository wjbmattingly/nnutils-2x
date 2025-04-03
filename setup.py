import os
from setuptools import setup, find_packages

# Get the long description from the README file
long_description = """
nnutils-2x: Neural network utilities for PyTorch 2.x

A collection of neural network utilities compatible with PyTorch 2.x.
"""

setup(
    name="nnutils-2x",
    version="0.1.0",
    description="Neural network utilities compatible with PyTorch 2.x",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/nnutils-2x",  # Replace with actual URL
    author="nnutils-2x Authors",
    author_email="author@example.com",  # Replace with actual email
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="pytorch, neural networks, deep learning",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=22.0.0",
            "isort>=5.0.0",
        ],
        "test": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/username/nnutils-2x/issues",
        "Source": "https://github.com/username/nnutils-2x",
    },
)
