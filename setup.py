"""Setup script for BIGCLAM project."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="bigclam-breast-cancer",
    version="1.0.0",
    author="Saeed Samimi",
    author_email="saeedsamimi995@gmail.com",
    description="BIGCLAM-based clustering of breast cancer subtypes from gene expression data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saeedsamimi995/New-Clustering-of-Breast-Cancer-Subtypes-Using-Gene-Expression-Data-and-the-BIGCLAM-Method",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "networkx>=3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bigclam-pipeline=main:main",
        ],
    },
)

