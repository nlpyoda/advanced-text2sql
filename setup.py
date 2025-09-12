#!/usr/bin/env python3
"""
Setup script for Advanced Text2SQL System
==========================================

Install the advanced text2sql package for easy usage and distribution.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
try:
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    # Fallback requirements
    requirements = [
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "datasets>=2.16.0",
        "accelerate>=0.25.0",
        "peft>=0.8.0",
        "trl>=0.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "sqlparse>=0.4.4",
        "wandb>=0.16.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0"
    ]

setup(
    name="advanced-text2sql",
    version="1.0.0",
    author="Advanced Text2SQL Team",
    author_email="contact@advanced-text2sql.com",
    description="Advanced Text2SQL System that outperforms Arctic-Text2SQL-R1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/advanced-text2sql",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Database",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
            "jupyter>=1.0.0",
        ],
        "gpu": [
            "bitsandbytes>=0.41.0",
            "deepspeed>=0.12.0",
            "flash-attn>=2.4.0",
        ],
        "evaluation": [
            "rouge-score>=0.1.2",
            "evaluate>=0.4.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "advanced-text2sql-train=run_complete_training:main",
            "advanced-text2sql-data=download_datasets:main",
            "advanced-text2sql-eval=run_training:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.json", "*.yaml", "*.yml"],
    },
    keywords=[
        "text2sql",
        "natural language processing",
        "database query",
        "machine learning",
        "transformers",
        "reinforcement learning",
        "schema understanding",
        "policy optimization"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/advanced-text2sql/issues",
        "Source": "https://github.com/your-username/advanced-text2sql",
        "Documentation": "https://github.com/your-username/advanced-text2sql/wiki",
    },
)