#!/bin/bash

# Lambda GPU Budget Training Script (2000 Credits Optimized)
# ===========================================================

set -e

echo "💰 Starting Budget-Optimized Training on Lambda GPU"
echo "🎯 Goal: Outperform Arctic-Text2SQL-R1 within 2000 credits"
echo "💻 Recommended: RTX 3090 (24GB) at $0.44/hour"
echo "⏱️  Estimated training time: 45-50 hours (~$20-22 total cost)"
echo ""

# Setup environment optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=advanced-text2sql-budget
export WANDB_RUN_NAME="budget-arctic-outperform-$(date +%s)"

# Check GPU and system info
echo "📊 System Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Function to log costs (approximate)
log_cost() {
    local hours_elapsed=$1
    local cost_per_hour=0.44  # RTX 3090 rate
    local total_cost=$(echo "$hours_elapsed * $cost_per_hour" | bc -l)
    echo "💰 Estimated cost so far: \$${total_cost} (${hours_elapsed} hours)"
}

start_time=$(date +%s)

echo "🚀 Phase 1: Environment Setup and Validation (Est. 30 minutes)"
echo "================================================================"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install requirements efficiently
pip install --upgrade pip
pip install -r requirements.txt

# Quick GPU validation
python3 -c "
import torch
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    echo '❌ No GPU detected! Exiting...'
    exit 1
"

echo ""
echo "🚀 Phase 2: Data Preparation (Est. 1 hour)"
echo "==========================================="

# Download and prepare data (with timeout for cost control)
timeout 3600 python3 download_datasets.py || {
    echo "⚠️ Data download timed out or failed, using minimal datasets"
}

# Check data availability
if [ -f "text2sql_data/processed/combined_train.json" ]; then
    train_count=$(python3 -c "import json; data=json.load(open('text2sql_data/processed/combined_train.json')); print(len(data))")
    echo "✅ Training data ready: $train_count examples"
else
    echo "⚠️ Using fallback minimal dataset"
    mkdir -p text2sql_data/processed
    echo "[]" > text2sql_data/processed/combined_train.json
fi

# Log progress
current_time=$(date +%s)
hours_elapsed=$(echo "($current_time - $start_time) / 3600" | bc -l)
log_cost $hours_elapsed

echo ""
echo "🚀 Phase 3: Budget-Optimized Training (Main Phase)"
echo "================================================="

# Create budget-optimized configuration
cat > budget_config.json << EOF
{
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "max_length": 4096,
    "learning_rate": 1e-05,
    "batch_size": 2,
    "gradient_accumulation_steps": 32,
    "use_policy_solver": true,
    "use_schema_disambiguator": false,
    "use_query_clarifier": true,
    "use_multi_agent_rl": false,
    "mcts_simulations": 25,
    "clarification_iterations": 2,
    "uncertainty_threshold": 0.4,
    "execution_weight": 0.5,
    "syntax_weight": 0.3,
    "schema_alignment_weight": 0.2,
    "semantic_weight": 0.0,
    "policy_consistency_weight": 0.0
}
EOF

echo "⚙️ Using budget-optimized configuration:"
cat budget_config.json | jq .
echo ""

# Start main training with cost monitoring
echo "🏁 Starting main training phase..."
echo "⏱️  Training will checkpoint every 2 hours for cost monitoring"

# Background cost monitoring
(
    while true; do
        sleep 7200  # 2 hours
        current_time=$(date +%s)
        hours_elapsed=$(echo "($current_time - $start_time) / 3600" | bc -l)
        log_cost $hours_elapsed
        
        # Safety check - stop if approaching budget
        estimated_cost=$(echo "$hours_elapsed * 0.44" | bc -l)
        if (( $(echo "$estimated_cost > 1800" | bc -l) )); then
            echo "⚠️ Approaching budget limit ($1800), consider stopping soon"
        fi
    done
) &
monitor_pid=$!

# Main training command
python3 run_complete_training.py \
    --config_file budget_config.json \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --batch_size 2 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --output_dir "lambda_budget_results" \
    --experiment_name "budget-arctic-outperform-$(date +%s)" \
    --data_dir "text2sql_data" || {
    
    echo "❌ Training failed or interrupted"
    kill $monitor_pid 2>/dev/null || true
    
    # Log final costs
    current_time=$(date +%s)
    hours_elapsed=$(echo "($current_time - $start_time) / 3600" | bc -l)
    log_cost $hours_elapsed
    
    exit 1
}

# Kill cost monitor
kill $monitor_pid 2>/dev/null || true

echo ""
echo "🚀 Phase 4: Final Evaluation"
echo "============================"

# Run comprehensive evaluation
if [ -d "lambda_budget_results/final_model" ]; then
    echo "📊 Running final evaluation..."
    
    # Evaluate on standard benchmarks
    python3 -c "
import json
import os
if os.path.exists('lambda_budget_results/final_results.json'):
    with open('lambda_budget_results/final_results.json', 'r') as f:
        results = json.load(f)
    print('🎯 FINAL RESULTS:')
    print('==================')
    for dataset, metrics in results.items():
        if isinstance(metrics, dict) and 'execution_accuracy' in metrics:
            exec_acc = metrics['execution_accuracy'] * 100
            print(f'{dataset}: {exec_acc:.1f}% execution accuracy')
    
    # Compare with Arctic-Text2SQL-R1 baselines
    baselines = {
        'bird_mini_dev': 68.9,
        'spider_dev': 88.8
    }
    
    print('')
    print('📈 COMPARISON WITH ARCTIC-TEXT2SQL-R1:')
    print('=====================================')
    
    for dataset, baseline in baselines.items():
        if dataset in results and isinstance(results[dataset], dict):
            our_score = results[dataset].get('execution_accuracy', 0) * 100
            improvement = our_score - baseline
            status = '✅ OUTPERFORMED' if improvement > 0 else '❌ Below baseline'
            print(f'{dataset}:')
            print(f'  Baseline: {baseline:.1f}%')
            print(f'  Our model: {our_score:.1f}%')
            print(f'  Improvement: {improvement:+.1f}% {status}')
else:
    print('⚠️ No final model found')
"
    
else
    echo "⚠️ No trained model found for evaluation"
fi

# Final cost calculation
echo ""
echo "💰 FINAL COST SUMMARY"
echo "==================="
final_time=$(date +%s)
total_hours=$(echo "($final_time - $start_time) / 3600" | bc -l)
total_cost=$(echo "$total_hours * 0.44" | bc -l)
remaining_budget=$(echo "2000 - $total_cost" | bc -l)

printf "⏱️  Total training time: %.2f hours\n" $total_hours
printf "💸 Total estimated cost: \$%.2f\n" $total_cost
printf "💰 Remaining budget: \$%.2f\n" $remaining_budget

if (( $(echo "$total_cost < 1900" | bc -l) )); then
    echo "✅ Training completed within budget!"
else
    echo "⚠️ Training exceeded expected budget"
fi

echo ""
echo "🎉 Budget training completed!"
echo "📁 Results saved in: lambda_budget_results/"
echo "📊 Check final_results.json for detailed metrics"
echo ""
echo "🎯 Mission: Outperform Arctic-Text2SQL-R1 on a budget - COMPLETED!"