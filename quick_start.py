#!/usr/bin/env python3
"""
Quick Start Script for Advanced Text2SQL Training
=================================================

This script provides a simplified interface to start training immediately.
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ is required")
        return False
    return True

def install_package():
    """Install the package in development mode."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        logger.info("Package installed successfully")
        return True
    except subprocess.CalledProcessError:
        logger.error("Package installation failed")
        return False

def run_quick_training():
    """Run training with optimized settings for quick results."""
    
    logger.info("Starting quick training run...")
    
    cmd = [
        sys.executable, "run_complete_training.py",
        "--model_name", "Qwen/Qwen2.5-7B-Instruct",
        "--batch_size", "1",  # Smaller for compatibility
        "--gradient_accumulation_steps", "32",
        "--learning_rate", "1e-5",
        "--experiment_name", "quick_test",
        "--output_dir", "quick_results"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        logger.info("Quick training completed!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        return False

def main():
    """Main quick start function."""
    
    print("🚀 Advanced Text2SQL Quick Start")
    print("=" * 40)
    
    # Check requirements
    if not check_python_version():
        return False
    
    # Install package
    if not install_package():
        return False
    
    # Run training
    success = run_quick_training()
    
    if success:
        print("\n✅ Quick start completed successfully!")
        print("Check 'quick_results/' directory for outputs.")
    else:
        print("\n❌ Quick start failed.")
        print("Check the logs above for details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)