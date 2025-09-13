#!/bin/bash

# Lambda GPU Setup Script for Advanced Text2SQL Training
# =====================================================

set -e

echo "🚀 Setting up Advanced Text2SQL on Lambda GPU..."

# System info
echo "📊 System Information:"
nvidia-smi
echo ""
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits
echo ""

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y git wget curl build-essential

# Install Python 3.10+ if not available
echo "🐍 Setting up Python environment..."
python3 --version || {
    echo "Installing Python 3.10..."
    sudo apt-get install -y python3.10 python3.10-pip python3.10-venv
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
}

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Clone repository
echo "📥 Cloning advanced-text2sql repository..."
if [ ! -d "advanced-text2sql" ]; then
    git clone https://github.com/nlpyoda/advanced-text2sql.git
fi
cd advanced-text2sql

# Install dependencies
echo "📋 Installing dependencies..."
pip install -r requirements.txt

# Additional GPU-optimized packages
echo "⚡ Installing GPU-optimized packages..."
pip install torch==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install flash-attn>=2.4.0 --no-build-isolation
pip install bitsandbytes>=0.41.0
pip install deepspeed>=0.12.0

# Verify GPU setup
echo "✅ Verifying GPU setup..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Set environment variables for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=advanced-text2sql-lambda

echo "🎉 Setup complete! Ready to train."
echo ""
echo "🚀 To start training:"
echo "  source venv/bin/activate"
echo "  cd advanced-text2sql"
echo "  python run_complete_training.py --batch_size 4 --gradient_accumulation_steps 8"
echo ""