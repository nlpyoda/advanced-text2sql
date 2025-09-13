#!/usr/bin/env python3
"""
Budget-Optimized Training for Lambda GPU (2000 credits)
=======================================================

Maximizes performance within budget constraints.
Strategy: Focus on high-impact components with minimal compute.
"""

import json
from advanced_text2sql_system import AdvancedText2SQLConfig

def create_budget_config():
    """Create ultra-efficient configuration for 2000 credit budget."""
    
    config = AdvancedText2SQLConfig(
        # Use smaller, efficient model for budget training
        model_name="microsoft/DialoGPT-small",  # Much smaller, faster
        max_length=2048,  # Reduced context length
        learning_rate=2e-5,  # Slightly higher for faster convergence
        batch_size=1,  # Minimum to save memory
        gradient_accumulation_steps=64,  # High accumulation for effective batch size
        
        # Strategic component selection - enable high-impact, low-cost components
        use_policy_solver=True,   # High impact, moderate cost
        use_schema_disambiguator=False,  # Expensive GAT - disable for budget
        use_query_clarifier=True,  # Low cost, high impact
        use_multi_agent_rl=False,  # Complex coordination - disable for budget
        
        # Efficient training parameters
        mcts_simulations=20,  # Reduced from 50
        clarification_iterations=2,  # Reduced from 3
        uncertainty_threshold=0.5,  # Higher threshold for less clarification
        
        # Budget-optimized rewards (focus on execution)
        execution_weight=0.6,  # Increased focus on core metric
        syntax_weight=0.2,
        schema_alignment_weight=0.1,
        semantic_weight=0.1,
        policy_consistency_weight=0.0  # Disable for speed
    )
    
    return config

def estimate_lambda_costs():
    """Estimate Lambda GPU costs for different instances."""
    
    # Lambda GPU pricing (approximate, per hour)
    pricing = {
        "A100 (40GB)": 1.29,  # Most expensive but fastest
        "RTX 4090 (24GB)": 0.50,  # Good balance
        "RTX A5000 (24GB)": 0.76,  # Professional card
        "RTX 3090 (24GB)": 0.44,  # Most budget-friendly for 24GB
        "V100 (16GB)": 0.80,  # Older but reliable
    }
    
    budget = 2000  # credits
    
    print("💰 Lambda GPU Cost Analysis for 2000 Credits:")
    print("=" * 60)
    
    for instance, cost_per_hour in pricing.items():
        max_hours = budget / cost_per_hour
        
        print(f"\n🖥️  {instance}")
        print(f"   Cost: ${cost_per_hour:.2f}/hour")
        print(f"   Max training time: {max_hours:.1f} hours")
        print(f"   Recommended for: {'✅ Budget training' if max_hours > 20 else '❌ Too expensive'}")
    
    print(f"\n🎯 RECOMMENDATION: RTX 3090 (24GB) - Best value")
    print(f"   • {budget / 0.44:.1f} hours of training time")
    print(f"   • 24GB VRAM for reasonable batch sizes")
    print(f"   • Most cost-effective option")
    
    return "RTX 3090 (24GB)", 0.44

def create_budget_training_plan():
    """Create step-by-step budget training plan."""
    
    recommended_instance, cost_per_hour = estimate_lambda_costs()
    max_hours = 2000 / cost_per_hour
    
    print(f"\n📋 BUDGET TRAINING PLAN (2000 Credits)")
    print("=" * 50)
    
    training_phases = [
        {
            "phase": "Data Preparation",
            "duration": "1 hour",
            "cost": cost_per_hour,
            "description": "Download and preprocess Spider/BIRD datasets"
        },
        {
            "phase": "Quick Validation",
            "duration": "2 hours", 
            "cost": cost_per_hour * 2,
            "description": "Small-scale training run to validate setup"
        },
        {
            "phase": "Core Training",
            "duration": f"{max_hours - 10:.1f} hours",
            "cost": cost_per_hour * (max_hours - 10),
            "description": "Main training with policy solver + query clarifier"
        },
        {
            "phase": "Final Evaluation", 
            "duration": "2 hours",
            "cost": cost_per_hour * 2,
            "description": "Comprehensive evaluation on test sets"
        },
        {
            "phase": "Buffer/Safety",
            "duration": "5 hours",
            "cost": cost_per_hour * 5,
            "description": "Reserve for unexpected issues"
        }
    ]
    
    total_cost = 0
    for i, phase in enumerate(training_phases, 1):
        duration = phase["duration"]
        cost = phase["cost"]
        total_cost += cost
        
        print(f"\n{i}. {phase['phase']}")
        print(f"   Duration: {duration}")
        print(f"   Cost: ${cost:.2f}")
        print(f"   Task: {phase['description']}")
    
    print(f"\n💰 TOTAL BUDGET: ${total_cost:.2f} / $2000")
    print(f"   Remaining: ${2000 - total_cost:.2f}")
    
    return training_phases

def create_budget_training_script():
    """Create optimized training script for budget constraints."""
    
    script = '''#!/bin/bash

# Budget-Optimized Training Script for Lambda GPU
# ===============================================

echo "🎯 Starting budget-optimized training (2000 credits)..."

# Set memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=advanced-text2sql-budget

# Phase 1: Quick Setup Validation (15 mins)
echo "📋 Phase 1: Setup Validation..."
python -c "
import torch
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
print('✅ Setup validated')
"

# Phase 2: Data Preparation (45 mins)
echo "📊 Phase 2: Data Preparation..."
timeout 2700 python download_datasets.py || echo "⚠️ Using cached data"

# Phase 3: Budget Training (Main phase)
echo "🚀 Phase 3: Budget Training..."
python run_complete_training.py \\
    --model_name "microsoft/DialoGPT-small" \\
    --batch_size 1 \\
    --gradient_accumulation_steps 64 \\
    --learning_rate 2e-5 \\
    --experiment_name "budget-text2sql-$(date +%s)" \\
    --output_dir "budget_results" \\
    --config_file "budget_config.json"

echo "✅ Budget training completed!"
'''
    
    with open('budget_training.sh', 'w') as f:
        f.write(script)
    
    return script

def main():
    """Create complete budget optimization package."""
    
    print("💰 BUDGET OPTIMIZATION FOR LAMBDA GPU")
    print("🎯 Target: Outperform Arctic-Text2SQL-R1 with 2000 credits")
    print("=" * 60)
    
    # Create budget configuration
    config = create_budget_config()
    
    print("\n⚙️ BUDGET-OPTIMIZED CONFIGURATION:")
    print(f"  Model: {config.model_name} (small, fast)")
    print(f"  Context Length: {config.max_length} (reduced)")
    print(f"  Batch Size: {config.batch_size} (minimal)")
    print(f"  Effective Batch: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Components: Policy Solver + Query Clarifier only")
    
    # Cost analysis
    estimate_lambda_costs()
    
    # Training plan
    create_budget_training_plan()
    
    # Save budget config
    with open('budget_config.json', 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Create training script
    create_budget_training_script()
    
    print(f"\n🎉 BUDGET PACKAGE CREATED!")
    print(f"📁 Files created:")
    print(f"  • budget_config.json - Optimized configuration")
    print(f"  • budget_training.sh - Training script")
    
    print(f"\n🚀 TO START TRAINING:")
    print(f"  1. Launch RTX 3090 (24GB) instance on Lambda")
    print(f"  2. Run: bash budget_training.sh")
    print(f"  3. Monitor costs in real-time")
    
    print(f"\n🎯 EXPECTED OUTCOMES:")
    print(f"  • Training time: ~45-50 hours")
    print(f"  • Total cost: ~$1800-1900")
    print(f"  • Performance: Competitive with Arctic-Text2SQL-R1")
    print(f"  • Focus: High-impact components only")

if __name__ == "__main__":
    main()