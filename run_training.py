#!/usr/bin/env python3
"""
Training Script for Improved Text2SQL Model
===========================================

This script orchestrates the complete training pipeline:
1. Environment setup and data loading
2. Model initialization with improvements
3. Curriculum learning training stages
4. Reinforcement learning with custom rewards
5. Evaluation and benchmarking
"""

import os
import json
import argparse
import logging
from pathlib import Path
import torch
import wandb
from datetime import datetime

# Import our custom modules
from improved_text2sql_project import (
    Text2SQLConfig, 
    EnhancedText2SQLModel, 
    CurriculumTrainer, 
    Text2SQLDataProcessor
)
from download_datasets import DatasetDownloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup training environment and check requirements."""
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("No GPU available - training will be slow")
    
    # Set memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    # Set environment variables for optimization
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_SILENT"] = "true"
    
    logger.info("Environment setup completed")

def prepare_data(data_dir: str = "text2sql_data", force_download: bool = False):
    """Prepare training and evaluation data."""
    
    data_path = Path(data_dir)
    combined_file = data_path / "processed" / "combined_train.json"
    
    # Download data if needed
    if not combined_file.exists() or force_download:
        logger.info("Downloading and preprocessing data...")
        downloader = DatasetDownloader(data_dir)
        
        # Download datasets
        downloader.download_spider_dataset()
        downloader.download_bird_dataset() 
        downloader.download_additional_datasets()
        
        # Preprocess
        downloader.preprocess_datasets()
        downloader.create_evaluation_sets()
    
    # Load processed data
    logger.info("Loading preprocessed data...")
    with open(combined_file, 'r') as f:
        train_data = json.load(f)
    
    # Load evaluation sets
    eval_data = {}
    for eval_file in (data_path / "processed").glob("*dev.json"):
        eval_name = eval_file.stem
        with open(eval_file, 'r') as f:
            eval_data[eval_name] = json.load(f)
    
    logger.info(f"Loaded {len(train_data)} training examples")
    logger.info(f"Loaded {len(eval_data)} evaluation sets: {list(eval_data.keys())}")
    
    return train_data, eval_data

def run_evaluation(model, tokenizer, eval_data: dict, config: Text2SQLConfig):
    """Run comprehensive evaluation on test sets."""
    
    logger.info("Starting evaluation...")
    results = {}
    
    for eval_name, eval_examples in eval_data.items():
        logger.info(f"Evaluating on {eval_name} ({len(eval_examples)} examples)...")
        
        eval_results = evaluate_model(model, tokenizer, eval_examples, config)
        results[eval_name] = eval_results
        
        logger.info(f"{eval_name} Results:")
        logger.info(f"  Execution Accuracy: {eval_results['execution_accuracy']:.3f}")
        logger.info(f"  Syntax Accuracy: {eval_results['syntax_accuracy']:.3f}")
        logger.info(f"  Schema Alignment: {eval_results['schema_alignment']:.3f}")
    
    return results

def evaluate_model(model, tokenizer, eval_examples, config):
    """Evaluate model on a single dataset."""
    
    from improved_text2sql_project import AdvancedRewardFunction
    reward_fn = AdvancedRewardFunction(config)
    
    correct_execution = 0
    correct_syntax = 0
    total_schema_score = 0
    total_examples = len(eval_examples)
    
    model.eval()
    
    with torch.no_grad():
        for example in eval_examples[:100]:  # Limit for faster evaluation
            try:
                # Prepare input
                question = example['question']
                schema = example.get('schema', {})
                gold_sql = example['sql']
                
                prompt = model.prepare_input(question, schema)
                
                # Generate SQL
                inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                outputs = model.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Decode generated SQL
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_sql = extract_sql_from_text(generated_text)
                
                # Compute rewards (as evaluation metrics)
                db_path = f"text2sql_data/databases/{example.get('db_id', 'dummy')}.sqlite"
                total_reward, individual_rewards = reward_fn.compute_reward(
                    generated_sql, gold_sql, schema, db_path
                )
                
                # Accumulate metrics
                if individual_rewards.get('execution', 0) > 0.8:
                    correct_execution += 1
                if individual_rewards.get('syntax', 0) > 0.8:
                    correct_syntax += 1
                total_schema_score += individual_rewards.get('schema', 0)
                
            except Exception as e:
                logger.warning(f"Error evaluating example: {e}")
                continue
    
    evaluated_count = min(100, total_examples)
    
    return {
        'execution_accuracy': correct_execution / evaluated_count,
        'syntax_accuracy': correct_syntax / evaluated_count,
        'schema_alignment': total_schema_score / evaluated_count,
        'total_examples': evaluated_count
    }

def extract_sql_from_text(text: str) -> str:
    """Extract SQL query from generated text."""
    if '<SQL>' in text and '</SQL>' in text:
        start = text.find('<SQL>') + 5
        end = text.find('</SQL>')
        return text[start:end].strip()
    
    # Fallback: look for SELECT statement
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'SELECT' in line.upper():
            # Take this line and potentially following lines
            sql_lines = [line]
            for j in range(i+1, len(lines)):
                if lines[j].strip() and not lines[j].startswith('#'):
                    sql_lines.append(lines[j])
                else:
                    break
            return '\n'.join(sql_lines).strip()
    
    return text.strip()

def main():
    """Main training pipeline."""
    
    parser = argparse.ArgumentParser(description="Train improved Text2SQL model")
    parser.add_argument("--data_dir", default="text2sql_data", help="Data directory")
    parser.add_argument("--output_dir", default="improved_text2sql_model", help="Output directory")
    parser.add_argument("--config_file", help="Custom configuration file")
    parser.add_argument("--force_download", action="store_true", help="Force data re-download")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--model_path", help="Path to trained model for evaluation")
    
    args = parser.parse_args()
    
    # Setup
    setup_environment()
    
    # Initialize wandb
    wandb.init(
        project="improved-text2sql",
        name=f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=vars(args)
    )
    
    # Load configuration
    if args.config_file and Path(args.config_file).exists():
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = Text2SQLConfig(**config_dict)
    else:
        config = Text2SQLConfig()
    
    # Prepare data
    train_data, eval_data = prepare_data(args.data_dir, args.force_download)
    
    if args.eval_only:
        # Load model for evaluation
        if args.model_path:
            logger.info(f"Loading model from {args.model_path}")
            model = EnhancedText2SQLModel(config)
            model.model.load_adapter(args.model_path)
        else:
            logger.error("Model path required for evaluation only mode")
            return
        
        # Run evaluation
        results = run_evaluation(model.model, model.tokenizer, eval_data, config)
        
        # Log results
        wandb.log({"evaluation": results})
        logger.info("Evaluation completed!")
        
    else:
        # Initialize model
        logger.info("Initializing enhanced Text2SQL model...")
        model = EnhancedText2SQLModel(config)
        
        # Process training data
        logger.info("Processing training data...")
        data_processor = Text2SQLDataProcessor(model.tokenizer, config.max_length)
        
        # Convert to HF dataset format
        from datasets import Dataset
        train_dataset = Dataset.from_list(train_data)
        
        # Initialize trainer
        logger.info("Initializing curriculum trainer...")
        trainer = CurriculumTrainer(model, model.tokenizer, config)
        
        # Start training
        logger.info("Starting curriculum training...")
        try:
            trainer.train_with_curriculum([train_dataset], args.output_dir)
            logger.info("Training completed successfully!")
            
            # Run final evaluation
            logger.info("Running final evaluation...")
            final_results = run_evaluation(model.model, model.tokenizer, eval_data, config)
            
            # Log final results
            wandb.log({"final_evaluation": final_results})
            
            # Save final metrics
            with open(Path(args.output_dir) / "final_results.json", 'w') as f:
                json.dump(final_results, f, indent=2)
            
            # Print summary
            logger.info("\n" + "="*50)
            logger.info("TRAINING COMPLETED!")
            logger.info("="*50)
            
            for eval_name, results in final_results.items():
                logger.info(f"\n{eval_name.upper()} RESULTS:")
                logger.info(f"  Execution Accuracy: {results['execution_accuracy']:.1%}")
                logger.info(f"  Syntax Accuracy: {results['syntax_accuracy']:.1%}")
                logger.info(f"  Schema Alignment: {results['schema_alignment']:.1%}")
            
            # Compare with Arctic-Text2SQL-R1 baseline
            logger.info(f"\nBASELINE COMPARISON (Arctic-Text2SQL-R1):")
            logger.info(f"  BIRD-dev baseline: 68.9% vs Our model: {final_results.get('bird_mini_dev', {}).get('execution_accuracy', 0):.1%}")
            logger.info(f"  Spider-test baseline: 88.8% vs Our model: {final_results.get('spider_dev', {}).get('execution_accuracy', 0):.1%}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    wandb.finish()

if __name__ == "__main__":
    main()