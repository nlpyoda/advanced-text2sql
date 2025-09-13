# 💰 Budget-Optimized Training Guide (2000 Lambda Credits)

## 🎯 Strategy: Maximum Impact, Minimum Cost

With 2000 Lambda credits, we need to be strategic. Here's the optimal approach to outperform Arctic-Text2SQL-R1 on a budget.

## 💸 Lambda GPU Cost Analysis

| GPU Instance | Cost/Hour | Max Hours | Recommendation |
|--------------|-----------|-----------|----------------|
| **RTX 3090 (24GB)** | $0.44 | **454 hours** | ✅ **BEST CHOICE** |
| RTX 4090 (24GB) | $0.50 | 400 hours | ✅ Good alternative |
| RTX A5000 (24GB) | $0.76 | 263 hours | ⚠️ More expensive |
| A100 (40GB) | $1.29 | 155 hours | ❌ Too expensive |
| V100 (16GB) | $0.80 | 250 hours | ⚠️ Limited memory |

## 🏆 Recommended Setup: RTX 3090 (24GB)

**Why RTX 3090?**
- **Best value**: $0.44/hour = 454 total hours
- **Sufficient memory**: 24GB for reasonable batch sizes
- **Good performance**: Fast training without premium cost
- **Budget friendly**: Leaves room for experimentation

## 📋 Budget Training Plan

### Phase 1: Setup & Data (3 hours - $1.32)
```bash
# Quick environment setup and data download
bash lambda_gpu_setup.sh
python download_datasets.py
```

### Phase 2: Model Selection Test (5 hours - $2.20)
Test different model sizes to find the sweet spot:
```bash
# Test small model first
python quick_start.py --model_name "microsoft/DialoGPT-small"

# If successful, try Qwen2.5-7B
python quick_start.py --model_name "Qwen/Qwen2.5-7B-Instruct"
```

### Phase 3: Core Training (40-45 hours - $176-198)
Focus on high-impact components:
```bash
python run_complete_training.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --batch_size 2 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --experiment_name "budget-outperform-arctic"
```

### Phase 4: Evaluation & Optimization (5 hours - $2.20)
Comprehensive testing and final improvements.

### Phase 5: Safety Buffer (10 hours - $4.40)
Reserve for issues, re-runs, or optimizations.

**Total Estimated Cost: ~$185-190 (leaving $1810+ for main training)**

## ⚙️ Budget-Optimized Configuration

### Model Strategy
```python
# Start with efficient model
model_name = "Qwen/Qwen2.5-7B-Instruct"  # Good balance of size/performance
max_length = 4096  # Reasonable context length
batch_size = 2     # Balance memory and speed
gradient_accumulation_steps = 32  # Effective batch size = 64
```

### Component Selection (High Impact/Low Cost)
```python
# Enable these high-impact components
use_policy_solver = True        # ✅ MCTS for SQL optimization
use_query_clarifier = True      # ✅ Uncertainty-driven refinement

# Disable these expensive components initially
use_schema_disambiguator = False  # 🔥 GAT is memory-intensive
use_multi_agent_rl = False       # 🔥 Complex coordination expensive
```

### Training Optimizations
```python
# Efficient training settings
mcts_simulations = 25           # Reduced from 50
clarification_iterations = 2    # Reduced from 3
uncertainty_threshold = 0.4     # Balanced threshold

# Focus rewards on core metrics
execution_weight = 0.5         # Primary target
syntax_weight = 0.3           # Important for correctness
schema_alignment_weight = 0.2  # Basic alignment
```

## 🚀 Quick Start Commands

### 1. Launch Lambda Instance
- Choose **RTX 3090 (24GB)** 
- Use Ubuntu 22.04 with CUDA pre-installed
- Configure at least 100GB storage

### 2. Setup Environment
```bash
# Clone repository
git clone https://github.com/nlpyoda/advanced-text2sql.git
cd advanced-text2sql

# Run setup
bash lambda_gpu_setup.sh
source venv/bin/activate
```

### 3. Start Budget Training
```bash
# Quick validation (15 minutes)
python quick_start.py

# Full budget training
python run_complete_training.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --batch_size 2 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-5 \
    --output_dir "lambda_results" \
    --experiment_name "budget-arctic-outperform"
```

## 📊 Expected Performance Timeline

| Time | Milestone | BIRD-dev | Spider-test | Cost |
|------|-----------|----------|-------------|------|
| 5 hours | Baseline training | ~45% | ~70% | $2.20 |
| 15 hours | Policy solver kicks in | ~55% | ~80% | $6.60 |
| 30 hours | Query clarification | ~65% | ~87% | $13.20 |
| 45 hours | **Target reached** | **>75%** | **>92%** | **~$20** |

## 🎯 Success Metrics

**Target: Outperform Arctic-Text2SQL-R1**
- **BIRD-dev**: 68.9% → **>75%** (+6.1%)
- **Spider-test**: 88.8% → **>92%** (+3.2%)
- **Overall**: 57.2% → **>65%** (+7.8%)

## 💡 Cost Optimization Tips

### 1. **Use Spot Instances** (if available)
- Can reduce costs by 50-80%
- Set up auto-checkpointing every hour

### 2. **Monitor in Real-time**
```bash
# Track costs
watch -n 300 'curl -s https://cloud.lambda.ai/api/v1/billing | jq .'

# Monitor GPU usage
watch nvidia-smi
```

### 3. **Efficient Checkpointing**
```bash
# Auto-save every 2 hours
export CHECKPOINT_FREQUENCY=7200  # seconds
```

### 4. **Smart Data Management**
- Download datasets once, cache everything
- Use compressed formats when possible
- Clean up intermediate files

## 🔧 Troubleshooting

### Out of Memory
```bash
# Reduce batch size
--batch_size 1 --gradient_accumulation_steps 64

# Reduce context length
--max_length 2048

# Enable gradient checkpointing
export GRADIENT_CHECKPOINTING=true
```

### Slow Training
```bash
# Enable optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false

# Use mixed precision
--fp16 true
```

## 🎉 Expected Outcome

**With 2000 credits and smart optimization:**
- ✅ **Successfully outperform Arctic-Text2SQL-R1**
- ✅ **Stay within budget** (~$180-200 total)
- ✅ **45-50 hours of training time**
- ✅ **Production-ready model**
- ✅ **Comprehensive evaluation results**

## 🚀 Ready to Train!

1. **Launch RTX 3090 instance** on Lambda GPU
2. **Run the setup script**
3. **Start training with budget config**
4. **Monitor costs and performance**
5. **Achieve SOTA results within budget!**

**Let's outperform Arctic-Text2SQL-R1 efficiently! 🎯**