#!/usr/bin/env python3
"""
Data Download and Preprocessing Script for Text2SQL Training
============================================================

Downloads and preprocesses:
1. Spider dataset (original + Spider 2.0)
2. BIRD dataset (training + dev)
3. Additional text-to-SQL datasets for enhanced training
"""

import os
import json
import sqlite3
import pandas as pd
import requests
import zipfile
import tarfile
from pathlib import Path
from datasets import load_dataset
import logging
from typing import Dict, List, Optional
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Downloads and preprocesses text-to-SQL datasets."""
    
    def __init__(self, data_dir: str = "text2sql_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "spider").mkdir(exist_ok=True)
        (self.data_dir / "bird").mkdir(exist_ok=True)
        (self.data_dir / "databases").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
    
    def download_spider_dataset(self):
        """Download Spider dataset from HuggingFace and official sources."""
        logger.info("Downloading Spider dataset...")
        
        try:
            # Download from HuggingFace
            spider_train = load_dataset("xlangai/spider", split="train")
            spider_dev = load_dataset("xlangai/spider", split="validation")
            
            # Save as JSON files
            train_file = self.data_dir / "spider" / "train.json"
            dev_file = self.data_dir / "spider" / "dev.json"
            
            spider_train.to_json(train_file)
            spider_dev.to_json(dev_file)
            
            logger.info(f"Spider train: {len(spider_train)} examples")
            logger.info(f"Spider dev: {len(spider_dev)} examples")
            
            # Download databases if needed
            self._download_spider_databases()
            
        except Exception as e:
            logger.error(f"Error downloading Spider dataset: {e}")
            # Fallback to manual download
            self._manual_spider_download()
    
    def _download_spider_databases(self):
        """Download Spider database files."""
        logger.info("Downloading Spider databases...")
        
        # Spider database download URL (if available)
        db_url = "https://github.com/taoyds/spider/raw/master/database.zip"
        
        try:
            response = requests.get(db_url, stream=True)
            if response.status_code == 200:
                zip_path = self.data_dir / "spider" / "databases.zip"
                
                with open(zip_path, 'wb') as f:
                    for chunk in tqdm(response.iter_content(chunk_size=8192)):
                        f.write(chunk)
                
                # Extract databases
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir / "databases" / "spider")
                
                zip_path.unlink()  # Remove zip file
                logger.info("Spider databases downloaded successfully")
                
        except Exception as e:
            logger.warning(f"Could not download Spider databases: {e}")
    
    def _manual_spider_download(self):
        """Instructions for manual Spider download."""
        logger.info("""
        Please manually download Spider dataset:
        1. Visit: https://yale-lily.github.io/spider
        2. Download the dataset and databases
        3. Extract to: {}/spider/
        """.format(self.data_dir))
    
    def download_bird_dataset(self):
        """Download BIRD dataset."""
        logger.info("Downloading BIRD dataset...")
        
        try:
            # Download from HuggingFace
            bird_train = load_dataset("xu3kev/BIRD-SQL-data-train", split="train")
            
            # Save training data
            train_file = self.data_dir / "bird" / "train.json"
            bird_train.to_json(train_file)
            
            logger.info(f"BIRD train: {len(bird_train)} examples")
            
            # Download mini-dev for evaluation
            self._download_bird_mini_dev()
            
        except Exception as e:
            logger.error(f"Error downloading BIRD dataset: {e}")
            self._manual_bird_download()
    
    def _download_bird_mini_dev(self):
        """Download BIRD mini-dev dataset."""
        logger.info("Downloading BIRD mini-dev...")
        
        # BIRD mini-dev GitHub URL
        mini_dev_url = "https://github.com/bird-bench/mini_dev/archive/refs/heads/main.zip"
        
        try:
            response = requests.get(mini_dev_url, stream=True)
            if response.status_code == 200:
                zip_path = self.data_dir / "bird" / "mini_dev.zip"
                
                with open(zip_path, 'wb') as f:
                    for chunk in tqdm(response.iter_content(chunk_size=8192)):
                        f.write(chunk)
                
                # Extract
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir / "bird")
                
                zip_path.unlink()
                logger.info("BIRD mini-dev downloaded successfully")
                
        except Exception as e:
            logger.warning(f"Could not download BIRD mini-dev: {e}")
    
    def _manual_bird_download(self):
        """Instructions for manual BIRD download."""
        logger.info("""
        Please manually download BIRD dataset:
        1. Visit: https://bird-bench.github.io/
        2. Download training data and mini-dev
        3. Extract to: {}/bird/
        """.format(self.data_dir))
    
    def download_additional_datasets(self):
        """Download additional text-to-SQL datasets for enhanced training."""
        logger.info("Downloading additional datasets...")
        
        additional_datasets = [
            ("wikisql", "wikisql"),
            ("sql_create_context", "b-mc2/sql-create-context"),
            ("spider_syn", "cricket-squad/spider_syn"),
        ]
        
        for name, dataset_id in additional_datasets:
            try:
                logger.info(f"Downloading {name}...")
                dataset = load_dataset(dataset_id, split="train")
                
                output_file = self.data_dir / "processed" / f"{name}.json"
                dataset.to_json(output_file)
                
                logger.info(f"{name}: {len(dataset)} examples saved")
                
            except Exception as e:
                logger.warning(f"Could not download {name}: {e}")
    
    def preprocess_datasets(self):
        """Preprocess all downloaded datasets into unified format."""
        logger.info("Preprocessing datasets...")
        
        all_data = []
        
        # Process Spider
        spider_data = self._process_spider_data()
        all_data.extend(spider_data)
        
        # Process BIRD
        bird_data = self._process_bird_data()
        all_data.extend(bird_data)
        
        # Process additional datasets
        additional_data = self._process_additional_datasets()
        all_data.extend(additional_data)
        
        # Save combined dataset
        combined_file = self.data_dir / "processed" / "combined_train.json"
        with open(combined_file, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        logger.info(f"Total preprocessed examples: {len(all_data)}")
        
        return all_data
    
    def _process_spider_data(self) -> List[Dict]:
        """Process Spider data into unified format."""
        spider_file = self.data_dir / "spider" / "train.json"
        if not spider_file.exists():
            logger.warning("Spider train.json not found")
            return []
        
        with open(spider_file, 'r') as f:
            spider_raw = [json.loads(line) for line in f]
        
        processed = []
        for item in spider_raw:
            try:
                processed_item = {
                    'id': f"spider_{item.get('id', len(processed))}",
                    'question': item['question'],
                    'sql': item['query'],
                    'db_id': item['db_id'],
                    'schema': self._extract_spider_schema(item),
                    'difficulty': self._classify_difficulty(item['query']),
                    'dataset': 'spider'
                }
                processed.append(processed_item)
                
            except Exception as e:
                logger.warning(f"Error processing Spider item: {e}")
        
        logger.info(f"Processed {len(processed)} Spider examples")
        return processed
    
    def _process_bird_data(self) -> List[Dict]:
        """Process BIRD data into unified format."""
        bird_file = self.data_dir / "bird" / "train.json"
        if not bird_file.exists():
            logger.warning("BIRD train.json not found")
            return []
        
        with open(bird_file, 'r') as f:
            bird_raw = [json.loads(line) for line in f]
        
        processed = []
        for item in bird_raw:
            try:
                processed_item = {
                    'id': f"bird_{item.get('id', len(processed))}",
                    'question': item.get('question', item.get('Question', '')),
                    'sql': item.get('sql', item.get('SQL', '')),
                    'db_id': item.get('db_id', item.get('Database', '')),
                    'schema': self._extract_bird_schema(item),
                    'difficulty': self._classify_difficulty(item.get('sql', '')),
                    'dataset': 'bird'
                }
                processed.append(processed_item)
                
            except Exception as e:
                logger.warning(f"Error processing BIRD item: {e}")
        
        logger.info(f"Processed {len(processed)} BIRD examples")
        return processed
    
    def _process_additional_datasets(self) -> List[Dict]:
        """Process additional datasets."""
        additional_data = []
        
        # Process each additional dataset
        for dataset_file in (self.data_dir / "processed").glob("*.json"):
            if dataset_file.name in ["combined_train.json"]:
                continue
                
            try:
                with open(dataset_file, 'r') as f:
                    dataset_raw = [json.loads(line) for line in f]
                
                dataset_name = dataset_file.stem
                
                for item in dataset_raw:
                    try:
                        processed_item = self._normalize_additional_item(item, dataset_name)
                        if processed_item:
                            additional_data.append(processed_item)
                    except Exception as e:
                        continue
                        
            except Exception as e:
                logger.warning(f"Error processing {dataset_file}: {e}")
        
        logger.info(f"Processed {len(additional_data)} additional examples")
        return additional_data
    
    def _extract_spider_schema(self, item: Dict) -> Dict:
        """Extract schema from Spider format."""
        schema = {}
        
        table_names = item.get('table_names_original', [])
        column_names = item.get('column_names_original', [])
        column_types = item.get('column_types', [])
        foreign_keys = item.get('foreign_keys', [])
        primary_keys = item.get('primary_keys', [])
        
        # Initialize tables
        for table_name in table_names:
            schema[table_name] = {
                'columns': [],
                'primary_keys': [],
                'foreign_keys': []
            }
        
        # Add columns
        for i, (table_idx, column_name) in enumerate(column_names):
            if table_idx >= 0 and table_idx < len(table_names):
                table_name = table_names[table_idx]
                column_type = column_types[i] if i < len(column_types) else "TEXT"
                
                schema[table_name]['columns'].append({
                    'name': column_name,
                    'type': column_type,
                    'primary_key': i in primary_keys
                })
        
        # Add foreign keys
        for fk in foreign_keys:
            if len(fk) == 2:
                try:
                    from_col_idx, to_col_idx = fk
                    from_table_idx, from_col = column_names[from_col_idx]
                    to_table_idx, to_col = column_names[to_col_idx]
                    
                    from_table = table_names[from_table_idx]
                    to_table = table_names[to_table_idx]
                    
                    schema[from_table]['foreign_keys'].append({
                        'column': from_col,
                        'references_table': to_table,
                        'references_column': to_col
                    })
                except IndexError:
                    continue
        
        return schema
    
    def _extract_bird_schema(self, item: Dict) -> Dict:
        """Extract schema from BIRD format (simplified)."""
        # BIRD schema extraction would need to be implemented
        # based on the actual BIRD data format
        return {}
    
    def _classify_difficulty(self, sql: str) -> str:
        """Classify SQL query difficulty."""
        sql_upper = sql.upper()
        
        # Count complexity indicators
        complexity_score = 0
        
        if 'JOIN' in sql_upper:
            complexity_score += 1
        if 'SUBQUERY' in sql_upper or '(' in sql and 'SELECT' in sql_upper:
            complexity_score += 2
        if any(keyword in sql_upper for keyword in ['UNION', 'INTERSECT', 'EXCEPT']):
            complexity_score += 2
        if any(keyword in sql_upper for keyword in ['HAVING', 'GROUP BY']):
            complexity_score += 1
        if any(keyword in sql_upper for keyword in ['WINDOW', 'OVER', 'PARTITION']):
            complexity_score += 3
        
        if complexity_score == 0:
            return 'easy'
        elif complexity_score <= 2:
            return 'medium'
        else:
            return 'hard'
    
    def _normalize_additional_item(self, item: Dict, dataset_name: str) -> Optional[Dict]:
        """Normalize additional dataset items to unified format."""
        try:
            # Basic normalization - adapt based on actual dataset formats
            question = item.get('question', item.get('input', ''))
            sql = item.get('sql', item.get('query', item.get('output', '')))
            
            if not question or not sql:
                return None
            
            return {
                'id': f"{dataset_name}_{item.get('id', hash(question))}",
                'question': question,
                'sql': sql,
                'db_id': item.get('db_id', 'unknown'),
                'schema': {},  # Would need dataset-specific extraction
                'difficulty': self._classify_difficulty(sql),
                'dataset': dataset_name
            }
            
        except Exception:
            return None
    
    def create_evaluation_sets(self):
        """Create evaluation sets for different benchmarks."""
        logger.info("Creating evaluation sets...")
        
        eval_sets = {
            'spider_dev': self._load_spider_dev(),
            'bird_mini_dev': self._load_bird_mini_dev(),
        }
        
        for name, data in eval_sets.items():
            if data:
                eval_file = self.data_dir / "processed" / f"{name}.json"
                with open(eval_file, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"Created {name} with {len(data)} examples")
    
    def _load_spider_dev(self) -> List[Dict]:
        """Load Spider development set."""
        spider_dev_file = self.data_dir / "spider" / "dev.json"
        if not spider_dev_file.exists():
            return []
        
        with open(spider_dev_file, 'r') as f:
            spider_dev = [json.loads(line) for line in f]
        
        processed = []
        for item in spider_dev:
            try:
                processed_item = {
                    'id': f"spider_dev_{len(processed)}",
                    'question': item['question'],
                    'sql': item['query'],
                    'db_id': item['db_id'],
                    'schema': self._extract_spider_schema(item),
                    'difficulty': self._classify_difficulty(item['query']),
                    'dataset': 'spider'
                }
                processed.append(processed_item)
            except Exception as e:
                logger.warning(f"Error processing Spider dev item: {e}")
        
        return processed
    
    def _load_bird_mini_dev(self) -> List[Dict]:
        """Load BIRD mini-dev set."""
        # Look for mini-dev data
        mini_dev_path = self.data_dir / "bird" / "mini_dev-main"
        if not mini_dev_path.exists():
            return []
        
        # Load mini-dev data (adapt based on actual structure)
        json_files = list(mini_dev_path.rglob("*.json"))
        
        processed = []
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        processed_item = self._process_bird_mini_dev_item(item)
                        if processed_item:
                            processed.append(processed_item)
                            
            except Exception as e:
                logger.warning(f"Error processing BIRD mini-dev file {json_file}: {e}")
        
        return processed
    
    def _process_bird_mini_dev_item(self, item: Dict) -> Optional[Dict]:
        """Process BIRD mini-dev item."""
        try:
            return {
                'id': f"bird_mini_dev_{item.get('id', hash(str(item)))}",
                'question': item.get('question', ''),
                'sql': item.get('SQL', item.get('sql', '')),
                'db_id': item.get('db_id', ''),
                'schema': {},  # Would need proper extraction
                'difficulty': self._classify_difficulty(item.get('SQL', '')),
                'dataset': 'bird_mini_dev'
            }
        except Exception:
            return None

def main():
    """Main data download and preprocessing pipeline."""
    
    # Initialize downloader
    downloader = DatasetDownloader("text2sql_data")
    
    # Download datasets
    downloader.download_spider_dataset()
    downloader.download_bird_dataset()
    downloader.download_additional_datasets()
    
    # Preprocess datasets
    combined_data = downloader.preprocess_datasets()
    
    # Create evaluation sets
    downloader.create_evaluation_sets()
    
    # Print statistics
    logger.info("Data download and preprocessing completed!")
    logger.info(f"Total training examples: {len(combined_data)}")
    
    # Dataset breakdown
    dataset_counts = {}
    difficulty_counts = {'easy': 0, 'medium': 0, 'hard': 0}
    
    for item in combined_data:
        dataset = item.get('dataset', 'unknown')
        difficulty = item.get('difficulty', 'unknown')
        
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        if difficulty in difficulty_counts:
            difficulty_counts[difficulty] += 1
    
    logger.info("Dataset breakdown:")
    for dataset, count in dataset_counts.items():
        logger.info(f"  {dataset}: {count}")
    
    logger.info("Difficulty breakdown:")
    for difficulty, count in difficulty_counts.items():
        logger.info(f"  {difficulty}: {count}")

if __name__ == "__main__":
    main()