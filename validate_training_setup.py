#!/usr/bin/env python3
"""
Validation script to check if the training setup is working correctly
"""

import os
import sys
import torch
import subprocess
from pathlib import Path

def check_cuda():
    """Check CUDA availability and devices"""
    print("=== CUDA Check ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA not available")
        return False
    return True

def check_distributed():
    """Check distributed training capabilities"""
    print("\n=== Distributed Check ===")
    print(f"NCCL available: {torch.distributed.is_nccl_available()}")
    print(f"MPI available: {torch.distributed.is_mpi_available()}")
    return True

def check_imports():
    """Check if all required imports work"""
    print("\n=== Import Check ===")
    
    # Add Qwen2-VL-Finetune to path
    qwen_path = '/Users/akhouriabhinavaditya/Qwen2-VL-Finetune/src'
    if not os.path.exists(qwen_path):
        print(f"❌ Qwen2-VL-Finetune path not found: {qwen_path}")
        return False
    
    sys.path.append(qwen_path)
    
    try:
        # Test basic transformers imports
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        print("✅ Transformers imports successful")
        
        # Test Qwen2-VL-Finetune imports
        from train.trainer import QwenTrainer
        from train.data import make_supervised_data_module
        from train.params import DataArguments, ModelArguments, TrainingArguments
        print("✅ Qwen2-VL-Finetune imports successful")
        
        # Test Liger kernel
        from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
        print("✅ Liger kernel imports successful")
        
        # Test monkey patch
        from monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward
        print("✅ Monkey patch imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def check_model_access():
    """Check if we can access the model"""
    print("\n=== Model Access Check ===")
    try:
        from transformers import AutoProcessor
        
        # Test with smallest model first
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        print(f"Testing access to {model_id}...")
        
        processor = AutoProcessor.from_pretrained(model_id)
        print("✅ Model access successful")
        return True
        
    except Exception as e:
        print(f"❌ Model access failed: {e}")
        print("This might be due to:")
        print("  - Missing HuggingFace token")
        print("  - Network connectivity issues")
        print("  - Model not available")
        return False

def check_training_script():
    """Check if training script can be parsed"""
    print("\n=== Training Script Check ===")
    
    script_path = "/Users/akhouriabhinavaditya/train_qwen25.py"
    if not os.path.exists(script_path):
        print(f"❌ Training script not found: {script_path}")
        return False
    
    try:
        # Test argument parsing
        result = subprocess.run([
            sys.executable, script_path, "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Training script argument parsing successful")
            return True
        else:
            print(f"❌ Training script failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Training script timed out")
        return False
    except Exception as e:
        print(f"❌ Training script check failed: {e}")
        return False

def check_environment():
    """Check environment variables and settings"""
    print("\n=== Environment Check ===")
    
    # Check important environment variables
    important_vars = [
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "NCCL_DEBUG"
    ]
    
    for var in important_vars:
        value = os.environ.get(var, "Not set")
        print(f"  {var}: {value}")
    
    # Check Python version
    print(f"  Python version: {sys.version}")
    
    # Check PyTorch version
    print(f"  PyTorch version: {torch.__version__}")
    
    return True

def main():
    """Run all validation checks"""
    print("🔍 Validating Qwen2.5 Training Setup")
    print("=" * 50)
    
    checks = [
        ("CUDA", check_cuda),
        ("Distributed", check_distributed),
        ("Imports", check_imports),
        ("Model Access", check_model_access),
        ("Training Script", check_training_script),
        ("Environment", check_environment),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} check failed with exception: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:<15}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All checks passed! Training setup is ready.")
        print("\nNext steps:")
        print("1. Run a small test training job")
        print("2. Monitor GPU memory usage")
        print("3. Test distributed training with multiple GPUs")
    else:
        print("⚠️  Some checks failed. Please fix the issues above before training.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)