"""
ChemJEPA Setup

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Core dependencies
requirements = [
    "torch>=2.0.0",
    "torch-geometric>=2.3.0",
    "rdkit>=2023.9.1",
    "e3nn>=0.5.0",
    "numpy>=1.22.0",
    "pandas>=1.5.0",
    "tqdm>=4.65.0",
    "pyyaml>=6.0",
    "scipy>=1.10.0",
]

# Optional dependencies
extras = {
    "dev": [
        "pytest>=7.3.0",
        "black>=23.3.0",
        "flake8>=6.0.0",
    ],
    "wandb": [
        "wandb>=0.15.0",
    ],
    "viz": [
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
}

setup(
    name="chemjepa",
    version="0.1.0",
    description="Joint-Embedding Predictive Architecture for Open-World Chemistry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ChemJEPA",
    url="https://github.com/yourusername/chemjepa",
    packages=find_packages(exclude=["tests", "scripts", "data", "checkpoints"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require=extras,
    entry_points={
        "console_scripts": [
            "chemjepa-test=test_quick:main",
            "chemjepa-train=train_production:main",
        ],
    },
)
