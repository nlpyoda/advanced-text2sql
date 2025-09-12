#!/usr/bin/env python3
"""
Complete Training Orchestration Script
=====================================

This script orchestrates the complete training pipeline to outperform Arctic-Text2SQL-R1:

1. Environment setup and dependency installation
2. Data download and preprocessing
3. Advanced model training with all components
4. Comprehensive evaluation and benchmarking
5. Performance comparison with baseline

Target Performance:
- BIRD-dev: >75% (baseline: 68.9%)
- Spider-test: >92% (baseline: 88.8%)
- Overall average: >65% (baseline: 57.2%)
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
import torch
import wandb
from datetime import datetime
import argparse

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our modules
from download_datasets import DatasetDownloader
from integrated_training_pipeline import AdvancedText2SQLTrainer
from advanced_text2sql_system import AdvancedText2SQLConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """Check and setup the training environment."""
    
    logger.info("Checking training environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8+ required")
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Found {gpu_count} GPU(s), {gpu_memory:.1f}GB memory on GPU 0")
        
        if gpu_memory < 16:
            logger.warning("GPU memory < 16GB. Training may be slow or fail.")
    else:
        logger.warning("No GPU found. Training will be very slow.")
    
    # Check disk space
    free_space = get_free_disk_space(".")
    if free_space < 50:  # 50GB minimum
        logger.warning(f"Low disk space: {free_space:.1f}GB. May not be enough for datasets.")
    
    logger.info("Environment check completed")

def install_dependencies():
    """Install required dependencies."""
    
    logger.info("Installing dependencies...")
    
    # Install requirements
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            logger.info("Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            raise
    else:
        logger.warning("requirements.txt not found, installing core dependencies...")
        core_deps = [
            "torch>=2.1.0", "transformers>=4.36.0", "datasets>=2.16.0",
            "wandb", "pandas", "numpy", "sqlparse", "tqdm", "requests"
        ]
        for dep in core_deps:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)

def download_and_prepare_data(data_dir: str = "text2sql_data", force_download: bool = False):
    """Download and prepare all training data."""
    
    logger.info("Downloading and preparing data...")
    
    # Initialize downloader
    downloader = DatasetDownloader(data_dir)
    
    # Check if data already exists
    combined_file = Path(data_dir) / "processed" / "combined_train.json"
    if combined_file.exists() and not force_download:
        logger.info("Data already exists, skipping download")
        with open(combined_file, 'r') as f:
            train_data = json.load(f)
    else:
        logger.info("Downloading datasets...")
        
        # Download datasets
        downloader.download_spider_dataset()
        downloader.download_bird_dataset()
        downloader.download_additional_datasets()
        
        # Preprocess
        logger.info("Preprocessing datasets...")
        train_data = downloader.preprocess_datasets()
        downloader.create_evaluation_sets()
    
    # Load evaluation data
    eval_data = {}
    eval_dir = Path(data_dir) / "processed"
    for eval_file in eval_dir.glob("*dev.json"):
        eval_name = eval_file.stem
        with open(eval_file, 'r') as f:
            eval_data[eval_name] = json.load(f)
    
    logger.info(f"Data preparation completed:")
    logger.info(f"  Training examples: {len(train_data)}")
    logger.info(f"  Evaluation sets: {list(eval_data.keys())}")
    
    return train_data, eval_data

def create_training_config(args) -> AdvancedText2SQLConfig:
    """Create training configuration."""
    
    config = AdvancedText2SQLConfig(
        # Model configuration
        model_name=args.model_name,
        max_length=8192,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        # Advanced features
        use_policy_solver=True,
        use_schema_disambiguator=True,
        use_query_clarifier=True,
        use_multi_agent_rl=True,
        
        # Policy solver settings
        mcts_simulations=50,
        mcts_exploration=1.414,
        policy_temperature=0.8,
        
        # Schema disambiguator settings
        schema_gat_heads=8,
        schema_gat_layers=3,
        schema_embedding_dim=512,
        
        # Query clarifier settings
        clarification_iterations=3,
        uncertainty_threshold=0.3,
        
        # Multi-agent settings
        num_agents=3,
        cooperation_weight=0.2,
        
        # Enhanced rewards
        execution_weight=0.35,
        syntax_weight=0.15,
        schema_alignment_weight=0.25,
        semantic_weight=0.15,
        policy_consistency_weight=0.10
    )
    
    return config

def run_training(train_data, eval_data, config, output_dir):
    """Run the complete training pipeline."""
    
    logger.info("Starting advanced Text2SQL training...")
    
    # Initialize trainer
    trainer = AdvancedText2SQLTrainer(config)
    
    # Run training with all advanced components
    results = trainer.train_with_advanced_pipeline(
        train_data, eval_data, output_dir
    )
    
    return results

def compare_with_baseline(results):
    """Compare results with Arctic-Text2SQL-R1 baseline."""
    
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE COMPARISON WITH ARCTIC-TEXT2SQL-R1")
    logger.info("="*60)
    
    baselines = {
        'bird_mini_dev': {'baseline': 0.689, 'baseline_name': 'BIRD-dev'},
        'spider_dev': {'baseline': 0.888, 'baseline_name': 'Spider-test'},
        'overall_average': {'baseline': 0.572, 'baseline_name': 'Overall Average'}
    }
    
    improvements = []
    
    for eval_name, result_data in results.items():
        if eval_name in baselines:
            baseline_info = baselines[eval_name]
            baseline_score = baseline_info['baseline']
            baseline_name = baseline_info['baseline_name']
            
            # Use execution accuracy as primary metric
            our_score = result_data.get('execution_accuracy', 0.0)
            
            improvement = our_score - baseline_score
            improvement_pct = (improvement / baseline_score) * 100 if baseline_score > 0 else 0
            
            logger.info(f"\n{baseline_name}:")
            logger.info(f"  Baseline (Arctic-Text2SQL-R1): {baseline_score:.1%}")
            logger.info(f"  Our Model: {our_score:.1%}")
            logger.info(f"  Improvement: {improvement:+.1%} ({improvement_pct:+.1f}%)")
            
            if improvement > 0:
                logger.info(f"  ✅ OUTPERFORMED BASELINE!")
            else:
                logger.info(f"  ❌ Below baseline")
            
            improvements.append(improvement)
    
    # Overall assessment
    avg_improvement = sum(improvements) / len(improvements) if improvements else 0
    logger.info(f"\nOVERALL ASSESSMENT:")
    logger.info(f"  Average improvement: {avg_improvement:+.1%}")
    
    successful_improvements = sum(1 for imp in improvements if imp > 0)
    logger.info(f"  Benchmarks outperformed: {successful_improvements}/{len(improvements)}")
    
    if avg_improvement > 0 and successful_improvements >= len(improvements) // 2:
        logger.info("  🎉 SUCCESSFULLY OUTPERFORMED ARCTIC-TEXT2SQL-R1!")
    else:
        logger.info("  ⚠️  Mixed results. Further optimization needed.")
    
    return avg_improvement > 0

def save_final_results(results, output_dir, args):
    """Save final results and model artifacts."""
    
    results_dir = Path(output_dir) / "final_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(results_dir / "detailed_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save configuration
    config_dict = vars(args)
    with open(results_dir / "training_config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Create summary report
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model_name,
        "training_completed": True,
        "results_summary": {
            eval_name: {
                "execution_accuracy": data.get("execution_accuracy", 0),
                "syntax_accuracy": data.get("syntax_accuracy", 0),
                "schema_alignment": data.get("schema_alignment", 0)
            }
            for eval_name, data in results.items()
        }
    }
    
    with open(results_dir / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {results_dir}")

def get_free_disk_space(path):
    """Get free disk space in GB."""
    try:
        statvfs = os.statvfs(path)
        free_space = statvfs.f_frsize * statvfs.f_bavail / (1024**3)
        return free_space
    except:
        return float('inf')  # Assume enough space if check fails

def main():
    """Main orchestration function."""
    
    parser = argparse.ArgumentParser(description="Train advanced Text2SQL model to outperform Arctic-Text2SQL-R1")
    
    # Model and training arguments
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct", 
                       help="Base model name")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                       help="Gradient accumulation steps")
    
    # Data arguments
    parser.add_argument("--data_dir", default="text2sql_data",
                       help="Data directory")
    parser.add_argument("--force_download", action="store_true",
                       help="Force re-download of data")
    
    # Output arguments  
    parser.add_argument("--output_dir", default="advanced_text2sql_results",
                       help="Output directory")
    parser.add_argument("--experiment_name", default=None,
                       help="Experiment name for wandb")
    
    # Control arguments
    parser.add_argument("--skip_env_check", action="store_true",
                       help="Skip environment checks")
    parser.add_argument("--skip_install", action="store_true",
                       help="Skip dependency installation")
    parser.add_argument("--data_only", action="store_true",
                       help="Only download and prepare data")
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not args.experiment_name:
            args.experiment_name = f"advanced_text2sql_{timestamp}"
        
        # Initialize wandb
        wandb.init(
            project="advanced-text2sql-outperform",
            name=args.experiment_name,
            config=vars(args)
        )
        
        logger.info(f"Starting experiment: {args.experiment_name}")
        
        # Step 1: Environment setup
        if not args.skip_env_check:
            check_environment()
        
        # Step 2: Install dependencies
        if not args.skip_install:
            install_dependencies()
        
        # Step 3: Data preparation
        train_data, eval_data = download_and_prepare_data(
            args.data_dir, args.force_download
        )
        
        if args.data_only:
            logger.info("Data preparation completed. Exiting as requested.")
            return
        
        # Step 4: Create configuration
        config = create_training_config(args)
        
        # Step 5: Run training
        results = run_training(train_data, eval_data, config, args.output_dir)
        
        # Step 6: Compare with baseline
        success = compare_with_baseline(results)
        
        # Step 7: Save results
        save_final_results(results, args.output_dir, args)
        
        # Log to wandb
        wandb.log({"final_results": results})
        wandb.log({"outperformed_baseline": success})
        
        # Final message
        if success:
            logger.info("\n🎉 MISSION ACCOMPLISHED!")
            logger.info("Successfully outperformed Arctic-Text2SQL-R1!")
        else:
            logger.info("\n⚠️ MISSION INCOMPLETE")
            logger.info("Model trained but didn't consistently outperform baseline.")
            logger.info("Consider hyperparameter tuning or additional training.")
        
        wandb.finish()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        wandb.log({"training_failed": True, "error": str(e)})
        wandb.finish()
        raise

if __name__ == "__main__":
    main()