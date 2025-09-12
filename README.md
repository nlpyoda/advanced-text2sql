# Advanced Text2SQL System - Outperforming Arctic-Text2SQL-R1

This repository contains an advanced Text2SQL system designed to surpass the performance of Arctic-Text2SQL-R1 (Paper: https://arxiv.org/pdf/2505.20315).

## 🎯 Performance Targets

| Benchmark | Arctic-Text2SQL-R1 | Our Target | Status |
|-----------|-------------------|------------|---------|
| BIRD-dev | 68.9% | **>75%** | 🎯 Target |
| Spider-test | 88.8% | **>92%** | 🎯 Target |
| Overall Average | 57.2% | **>65%** | 🎯 Target |

## 🚀 Key Innovations

### 1. Advanced Training Methods
- **Policy Solvers**: Monte Carlo Tree Search (MCTS) for optimal SQL generation policies
- **Schema Disambiguators**: Graph Attention Networks (GAT) for schema understanding
- **Query Clarifiers**: Uncertainty-driven iterative refinement
- **Multi-Agent RL**: Coordinated specialized agents (Schema, SQL, Validation)

### 2. Enhanced Architecture
- **SQL-Aware Tokenization**: Specialized tokens for SQL components
- **Curriculum Learning**: Progressive difficulty training
- **Advanced Reward Functions**: Multi-objective optimization
- **Uncertainty Estimation**: Confidence-aware generation

### 3. Training Pipeline
- **Phase 1**: Curriculum learning with multi-agent coordination
- **Phase 2**: Policy-guided training with MCTS
- **Phase 3**: Schema-aware fine-tuning with GAT
- **Phase 4**: Uncertainty-driven refinement

## 📋 Requirements

### Hardware Requirements
- **GPU**: 16GB+ VRAM (RTX 3090/4090, V100, A100)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ free space
- **CPU**: 8+ cores recommended

### Software Requirements
- **Python**: 3.8+
- **CUDA**: 11.8+ (if using GPU)
- **Operating System**: Linux/macOS (Windows with WSL)

## 🛠️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/advanced-text2sql
cd advanced-text2sql
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Wandb (Optional)
```bash
wandb login
```

## 🚀 Quick Start

### Option 1: Full Training Pipeline (Recommended)
```bash
python run_complete_training.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --output_dir "results" \
    --experiment_name "text2sql_outperform"
```

### Option 2: Data Download Only
```bash
python run_complete_training.py --data_only
```

### Option 3: Step-by-Step Execution

#### Step 1: Download Data
```bash
python download_datasets.py
```

#### Step 2: Train with Advanced Pipeline
```bash
python integrated_training_pipeline.py
```

## 📊 Training Phases

### Phase 1: Curriculum Learning (2-3 hours)
- Progressive difficulty: Easy → Medium → Hard
- Multi-agent coordination for each example
- Schema-SQL alignment training

### Phase 2: Policy-Guided Training (3-4 hours)
- MCTS policy solver integration
- Exploration-exploitation balance
- Action space optimization

### Phase 3: Schema-Aware Training (2-3 hours)
- Graph Attention Network training
- Schema disambiguation learning
- Relationship understanding

### Phase 4: Uncertainty Refinement (1-2 hours)
- Query clarification training
- Uncertainty estimation calibration
- Iterative improvement

**Total Training Time: 8-12 hours on RTX 4090**

## 📁 Project Structure

```
advanced-text2sql/
├── README.md                           # This file
├── requirements.txt                    # Dependencies
├── run_complete_training.py           # Main orchestration script
├── download_datasets.py               # Data download/preprocessing
├── improved_text2sql_project.py      # Enhanced baseline model
├── advanced_text2sql_system.py       # Advanced components
├── integrated_training_pipeline.py   # Complete training pipeline
├── run_training.py                    # Alternative training script
├── text2sql_data/                     # Data directory
│   ├── spider/                        # Spider dataset
│   ├── bird/                          # BIRD dataset
│   ├── databases/                     # Database files
│   └── processed/                     # Processed datasets
└── results/                           # Training outputs
    ├── phase1/                        # Phase 1 checkpoints
    ├── phase2/                        # Phase 2 checkpoints
    ├── phase3/                        # Phase 3 checkpoints
    ├── phase4/                        # Phase 4 checkpoints
    └── final_model/                   # Final trained model
```

## ⚙️ Configuration Options

### Model Configuration
```python
config = AdvancedText2SQLConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",  # Base model
    max_length=8192,                          # Context length
    learning_rate=1e-5,                       # Learning rate
    batch_size=2,                             # Batch size
    gradient_accumulation_steps=16,           # Gradient accumulation
)
```

### Advanced Features
```python
# Enable/disable advanced components
use_policy_solver=True,              # MCTS policy solver
use_schema_disambiguator=True,       # GAT schema understanding
use_query_clarifier=True,            # Uncertainty-driven refinement
use_multi_agent_rl=True,            # Multi-agent coordination
```

### Training Parameters
```python
# MCTS Policy Solver
mcts_simulations=50,                 # MCTS simulation count
mcts_exploration=1.414,              # UCB exploration parameter
policy_temperature=0.8,              # Policy sampling temperature

# Schema Disambiguator
schema_gat_heads=8,                  # GAT attention heads
schema_gat_layers=3,                 # GAT layer count
schema_embedding_dim=512,            # Schema embedding dimension

# Query Clarifier
clarification_iterations=3,          # Max clarification rounds
uncertainty_threshold=0.3,           # Uncertainty threshold
```

## 📈 Monitoring Training

### Wandb Dashboard
- Real-time loss tracking
- Performance metrics
- Hyperparameter comparison
- System resource usage

### Local Monitoring
```bash
# Watch training logs
tail -f results/training.log

# Monitor GPU usage
nvidia-smi -l 1

# Check training progress
ls -la results/*/
```

## 🧪 Evaluation

### Automatic Evaluation
The training script automatically evaluates on:
- Spider development set
- BIRD mini-dev set
- Additional benchmarks

### Manual Evaluation
```python
from integrated_training_pipeline import AdvancedText2SQLTrainer
from advanced_text2sql_system import AdvancedText2SQLConfig

# Load trained model
config = AdvancedText2SQLConfig()
trainer = AdvancedText2SQLTrainer(config)

# Load model weights
trainer.base_model.load_pretrained("results/final_model/model")

# Run evaluation
results = trainer.comprehensive_evaluation(eval_data)
```

## 🎯 Expected Results

Based on our advanced techniques, we expect:

### Quantitative Improvements
- **BIRD-dev**: 68.9% → **76%+** (+7.1% absolute)
- **Spider-test**: 88.8% → **93%+** (+4.2% absolute)
- **Overall Average**: 57.2% → **67%+** (+9.8% absolute)

### Qualitative Improvements
- Better schema understanding
- More coherent SQL generation
- Improved handling of complex queries
- Higher confidence calibration

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
--batch_size 1 --gradient_accumulation_steps 32

# Use model sharding
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

#### 2. Dataset Download Failures
```bash
# Manual download
python download_datasets.py --force_download

# Check internet connection and disk space
df -h
```

#### 3. Training Convergence Issues
```bash
# Adjust learning rate
--learning_rate 5e-6

# Increase training time
# Add more curriculum stages
```

### Performance Optimization

#### For Limited GPU Memory
```python
# Use gradient checkpointing
config.use_gradient_checkpointing = True

# Reduce sequence length
config.max_length = 4096

# Use 8-bit quantization
config.load_in_8bit = True
```

#### For Faster Training
```python
# Use mixed precision
config.fp16 = True

# Increase batch size if possible
config.batch_size = 4

# Use multiple GPUs
config.num_gpus = 2
```

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{advanced-text2sql-2024,
  title={Advanced Text2SQL: Outperforming Arctic-Text2SQL-R1 with Policy Solvers and Schema Disambiguation},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/advanced-text2sql}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 🙋‍♂️ Support

- **Issues**: Open GitHub issue
- **Discussions**: GitHub discussions
- **Email**: your.email@example.com

## 🔮 Future Work

- [ ] Integration with more SQL databases
- [ ] Support for natural language explanations
- [ ] Real-time query optimization
- [ ] Multi-modal schema understanding
- [ ] Federated learning across databases

---

**Ready to outperform Arctic-Text2SQL-R1? Let's get started! 🚀**