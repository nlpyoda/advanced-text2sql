#!/usr/bin/env python3
"""
Lambda GPU Training Configuration for Advanced Text2SQL
=======================================================

Optimized configuration for training on Lambda GPU instances.
"""

import os
import json
from advanced_text2sql_system import AdvancedText2SQLConfig

def create_lambda_gpu_config(gpu_memory_gb=24, num_gpus=1):
    """Create optimized configuration for Lambda GPU training."""
    
    # Base configuration optimized for Lambda GPU
    if gpu_memory_gb >= 40:  # A100 40GB or similar
        config = AdvancedText2SQLConfig(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            max_length=8192,
            learning_rate=1e-5,
            batch_size=4,
            gradient_accumulation_steps=8,
            use_policy_solver=True,
            use_schema_disambiguator=True,
            use_query_clarifier=True,
            use_multi_agent_rl=True,
        )
    elif gpu_memory_gb >= 24:  # RTX 4090, RTX A5000, etc.
        config = AdvancedText2SQLConfig(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            max_length=6144,
            learning_rate=1e-5,
            batch_size=2,
            gradient_accumulation_steps=16,
            use_policy_solver=True,
            use_schema_disambiguator=True,
            use_query_clarifier=True,
            use_multi_agent_rl=True,
        )
    else:  # 16GB or less
        config = AdvancedText2SQLConfig(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            max_length=4096,
            learning_rate=1e-5,
            batch_size=1,
            gradient_accumulation_steps=32,
            use_policy_solver=True,
            use_schema_disambiguator=False,  # Disable for memory
            use_query_clarifier=True,
            use_multi_agent_rl=False,  # Disable for memory
        )
    
    return config

def get_lambda_gpu_info():
    """Get Lambda GPU instance information."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 2:
                    name = parts[0]
                    memory_mb = int(parts[1])
                    memory_gb = memory_mb / 1024
                    gpu_info.append({'name': name, 'memory_gb': memory_gb})
            
            return gpu_info
        
    except Exception as e:
        print(f"Could not detect GPU info: {e}")
    
    return [{'name': 'Unknown GPU', 'memory_gb': 24}]  # Default assumption

def main():
    """Create and display Lambda GPU configuration."""
    
    print("🔍 Detecting Lambda GPU configuration...")
    
    gpu_info = get_lambda_gpu_info()
    print(f"Found {len(gpu_info)} GPU(s):")
    
    for i, gpu in enumerate(gpu_info):
        print(f"  GPU {i}: {gpu['name']} ({gpu['memory_gb']:.1f} GB)")
    
    # Use the GPU with most memory
    max_memory = max(gpu['memory_gb'] for gpu in gpu_info)
    num_gpus = len(gpu_info)
    
    print(f"\n🎯 Optimizing for {max_memory:.1f} GB GPU memory...")
    
    # Create optimized configuration
    config = create_lambda_gpu_config(max_memory, num_gpus)
    
    print(f"\n⚙️ Lambda GPU Training Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Max Length: {config.max_length}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Learning Rate: {config.learning_rate}")
    
    print(f"\n🧠 Advanced Components:")
    print(f"  Policy Solver: {'✅' if config.use_policy_solver else '❌'}")
    print(f"  Schema Disambiguator: {'✅' if config.use_schema_disambiguator else '❌'}")
    print(f"  Query Clarifier: {'✅' if config.use_query_clarifier else '❌'}")
    print(f"  Multi-Agent RL: {'✅' if config.use_multi_agent_rl else '❌'}")
    
    # Save configuration
    config_dict = config.__dict__
    with open('lambda_gpu_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n💾 Configuration saved to: lambda_gpu_config.json")
    
    # Estimate training time
    estimated_examples = 10000  # Rough estimate
    examples_per_hour = config.batch_size * config.gradient_accumulation_steps * 200  # Conservative
    estimated_hours = estimated_examples / examples_per_hour
    
    print(f"\n⏱️ Estimated Training Time:")
    print(f"  Training Examples: ~{estimated_examples:,}")
    print(f"  Processing Rate: ~{examples_per_hour:,} examples/hour")
    print(f"  Estimated Duration: ~{estimated_hours:.1f} hours")
    
    print(f"\n🚀 Ready to start training!")
    print(f"  Run: python run_complete_training.py --config_file lambda_gpu_config.json")

if __name__ == "__main__":
    main()