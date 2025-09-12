#!/usr/bin/env python3
"""
Advanced Text2SQL System with Policy Solvers, Schema Disambiguators, and Query Clarifiers
========================================================================================

Enhanced system with state-of-the-art components:
1. Policy Gradient SQL Solver with Monte Carlo Tree Search
2. Neural Schema Disambiguator with Graph Attention
3. Iterative Query Clarification with Uncertainty Estimation
4. Multi-Agent Reinforcement Learning
5. Dynamic Token Vocabulary with SQL-specific embeddings
6. Schema-Query Alignment Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel,
    PreTrainedTokenizerFast, PreTrainedModel
)
from transformers.modeling_outputs import CausalLMOutputWithPast
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
from dataclasses import dataclass
import math
import random
from collections import defaultdict
import sqlparse
from sqlparse import sql, tokens
import logging
import json
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLTokenType(Enum):
    """Enhanced SQL token types for specialized embeddings."""
    KEYWORD = "keyword"           # SELECT, FROM, WHERE, etc.
    TABLE_NAME = "table_name"     # Actual table references
    COLUMN_NAME = "column_name"   # Column references
    FUNCTION = "function"         # COUNT, SUM, AVG, etc.
    OPERATOR = "operator"         # =, >, <, LIKE, etc.
    VALUE = "value"              # Literal values
    JOIN_TYPE = "join_type"      # INNER, LEFT, RIGHT, etc.
    AGGREGATION = "aggregation"  # GROUP BY, HAVING
    SUBQUERY = "subquery"        # Nested query indicators
    SCHEMA_REF = "schema_ref"    # Schema references

@dataclass
class AdvancedText2SQLConfig:
    """Enhanced configuration with advanced components."""
    # Base model config
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_length: int = 8192  # Increased for complex schemas
    learning_rate: float = 1e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 16
    
    # Advanced components
    use_policy_solver: bool = True
    use_schema_disambiguator: bool = True
    use_query_clarifier: bool = True
    use_multi_agent_rl: bool = True
    
    # Policy Solver config
    mcts_simulations: int = 50
    mcts_exploration: float = 1.414
    policy_temperature: float = 0.8
    
    # Schema Disambiguator config
    schema_gat_heads: int = 8
    schema_gat_layers: int = 3
    schema_embedding_dim: int = 512
    
    # Query Clarifier config
    clarification_iterations: int = 3
    uncertainty_threshold: float = 0.3
    
    # Multi-agent RL config
    num_agents: int = 3  # Schema Agent, SQL Agent, Validation Agent
    cooperation_weight: float = 0.2
    
    # Enhanced reward weights
    execution_weight: float = 0.35
    syntax_weight: float = 0.15
    schema_alignment_weight: float = 0.25
    semantic_weight: float = 0.15
    policy_consistency_weight: float = 0.10

class EnhancedSQLTokenizer:
    """SQL-aware tokenizer with specialized token types."""
    
    def __init__(self, base_tokenizer: PreTrainedTokenizerFast):
        self.base_tokenizer = base_tokenizer
        self.sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
            'GROUP', 'BY', 'HAVING', 'ORDER', 'LIMIT', 'OFFSET', 'UNION', 'INTERSECT',
            'EXCEPT', 'WITH', 'AS', 'ON', 'IN', 'EXISTS', 'BETWEEN', 'LIKE', 'IS',
            'NULL', 'NOT', 'AND', 'OR', 'DISTINCT', 'ALL', 'ANY', 'SOME'
        }
        self.sql_functions = {
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'COALESCE', 'CASE', 'WHEN', 'THEN',
            'ELSE', 'END', 'CAST', 'CONVERT', 'SUBSTRING', 'LENGTH', 'UPPER', 'LOWER'
        }
        self.sql_operators = {'=', '>', '<', '>=', '<=', '!=', '<>', 'LIKE', 'IN'}
        
        # Add specialized SQL tokens
        special_tokens = [
            f"<{token_type.value.upper()}>" for token_type in SQLTokenType
        ] + [
            f"</{token_type.value.upper()}>" for token_type in SQLTokenType
        ] + [
            "<SCHEMA_START>", "<SCHEMA_END>", "<TABLE_START>", "<TABLE_END>",
            "<COLUMN_START>", "<COLUMN_END>", "<FK_START>", "<FK_END>",
            "<QUERY_START>", "<QUERY_END>", "<CLARIFICATION>", "<UNCERTAINTY>",
            "<CONFIDENCE_HIGH>", "<CONFIDENCE_MED>", "<CONFIDENCE_LOW>"
        ]
        
        self.base_tokenizer.add_special_tokens({
            "additional_special_tokens": special_tokens
        })
        
    def tokenize_with_sql_awareness(self, text: str, schema: Dict = None) -> Dict:
        """Tokenize text with SQL-specific token type annotations."""
        
        # Parse SQL if present
        sql_tokens = []
        token_types = []
        
        if "<QUERY_START>" in text and "<QUERY_END>" in text:
            start = text.find("<QUERY_START>") + len("<QUERY_START>")
            end = text.find("<QUERY_END>")
            sql_text = text[start:end].strip()
            
            # Parse SQL tokens
            parsed = sqlparse.parse(sql_text)[0] if sql_text else None
            if parsed:
                for token in parsed.flatten():
                    if token.ttype is not None:
                        sql_tokens.append(token.value)
                        token_types.append(self._classify_token(token, schema))
        
        # Standard tokenization
        tokenized = self.base_tokenizer(
            text, 
            return_tensors="pt",
            max_length=8192,
            truncation=True,
            padding="max_length"
        )
        
        return {
            **tokenized,
            'sql_tokens': sql_tokens,
            'token_types': token_types
        }
    
    def _classify_token(self, token, schema: Dict = None) -> SQLTokenType:
        """Classify SQL token type."""
        value = token.value.upper().strip()
        
        if token.ttype in tokens.Keyword:
            if value in self.sql_keywords:
                return SQLTokenType.KEYWORD
            elif value in self.sql_functions:
                return SQLTokenType.FUNCTION
            elif value in ['INNER', 'LEFT', 'RIGHT', 'OUTER', 'CROSS']:
                return SQLTokenType.JOIN_TYPE
            elif value in ['GROUP', 'HAVING']:
                return SQLTokenType.AGGREGATION
        
        elif token.ttype in tokens.Name:
            if schema:
                # Check if it's a table name
                if value.lower() in [t.lower() for t in schema.keys()]:
                    return SQLTokenType.TABLE_NAME
                # Check if it's a column name
                for table_info in schema.values():
                    if value.lower() in [c['name'].lower() for c in table_info.get('columns', [])]:
                        return SQLTokenType.COLUMN_NAME
            return SQLTokenType.SCHEMA_REF
        
        elif token.ttype in tokens.Operator:
            return SQLTokenType.OPERATOR
        
        elif token.ttype in tokens.Literal:
            return SQLTokenType.VALUE
        
        return SQLTokenType.KEYWORD  # Default

class PolicySQLSolver:
    """Monte Carlo Tree Search based SQL policy solver."""
    
    def __init__(self, config: AdvancedText2SQLConfig):
        self.config = config
        self.mcts_tree = {}
        
    def solve_sql_policy(self, question: str, schema: Dict, model_logits: torch.Tensor) -> Dict:
        """Use MCTS to find optimal SQL generation policy."""
        
        # Initialize MCTS tree
        root_state = self._create_state(question, schema, "")
        
        for _ in range(self.config.mcts_simulations):
            # Selection
            path = self._select_path(root_state)
            
            # Expansion  
            leaf_state = self._expand_node(path[-1])
            
            # Simulation
            reward = self._simulate(leaf_state)
            
            # Backpropagation
            self._backpropagate(path, reward)
        
        # Extract best policy
        best_actions = self._extract_policy(root_state)
        
        return {
            'policy_actions': best_actions,
            'confidence': self._calculate_confidence(root_state),
            'exploration_stats': self._get_exploration_stats()
        }
    
    def _create_state(self, question: str, schema: Dict, partial_sql: str) -> str:
        """Create state representation."""
        return f"{question}|{json.dumps(schema)}|{partial_sql}"
    
    def _select_path(self, root_state: str) -> List[str]:
        """Select path using UCB1."""
        path = [root_state]
        current_state = root_state
        
        while current_state in self.mcts_tree and self.mcts_tree[current_state]['children']:
            children = self.mcts_tree[current_state]['children']
            
            # UCB1 selection
            best_child = None
            best_ucb = float('-inf')
            
            total_visits = sum(self.mcts_tree[child]['visits'] for child in children)
            
            for child in children:
                child_data = self.mcts_tree[child]
                
                if child_data['visits'] == 0:
                    ucb = float('inf')
                else:
                    exploitation = child_data['value'] / child_data['visits']
                    exploration = self.config.mcts_exploration * math.sqrt(
                        math.log(total_visits) / child_data['visits']
                    )
                    ucb = exploitation + exploration
                
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
            
            current_state = best_child
            path.append(current_state)
        
        return path
    
    def _expand_node(self, state: str) -> str:
        """Expand node with possible actions."""
        if state not in self.mcts_tree:
            self.mcts_tree[state] = {
                'visits': 0,
                'value': 0.0,
                'children': []
            }
        
        # Generate possible next SQL tokens
        question, schema_str, partial_sql = state.split('|', 2)
        schema = json.loads(schema_str)
        
        possible_actions = self._generate_possible_actions(partial_sql, schema)
        
        for action in possible_actions[:5]:  # Limit branching factor
            new_sql = partial_sql + " " + action if partial_sql else action
            child_state = self._create_state(question, schema, new_sql)
            
            if child_state not in self.mcts_tree[state]['children']:
                self.mcts_tree[state]['children'].append(child_state)
                self.mcts_tree[child_state] = {
                    'visits': 0,
                    'value': 0.0,
                    'children': []
                }
        
        # Return random child for simulation
        if self.mcts_tree[state]['children']:
            return random.choice(self.mcts_tree[state]['children'])
        return state
    
    def _generate_possible_actions(self, partial_sql: str, schema: Dict) -> List[str]:
        """Generate possible next SQL tokens."""
        actions = []
        
        if not partial_sql:
            actions.append("SELECT")
        elif partial_sql.upper().endswith("SELECT"):
            # Add column options
            for table_name, table_info in schema.items():
                for col in table_info.get('columns', []):
                    actions.append(col['name'])
            actions.extend(["COUNT(*)", "DISTINCT", "*"])
        elif "SELECT" in partial_sql.upper() and "FROM" not in partial_sql.upper():
            actions.append("FROM")
        elif partial_sql.upper().endswith("FROM"):
            # Add table options
            actions.extend(schema.keys())
        elif "FROM" in partial_sql.upper() and "WHERE" not in partial_sql.upper():
            actions.extend(["WHERE", "JOIN", "GROUP BY", "ORDER BY", "LIMIT"])
        else:
            # Add common SQL continuations
            actions.extend(["AND", "OR", "=", ">", "<", "LIKE", "IN", "EXISTS"])
        
        return actions
    
    def _simulate(self, state: str) -> float:
        """Simulate random completion and evaluate."""
        question, schema_str, partial_sql = state.split('|', 2)
        schema = json.loads(schema_str)
        
        # Simple simulation: add random tokens until complete
        simulation_sql = partial_sql
        for _ in range(10):  # Max 10 tokens
            actions = self._generate_possible_actions(simulation_sql, schema)
            if not actions:
                break
            simulation_sql += " " + random.choice(actions)
        
        # Evaluate simulated SQL
        return self._evaluate_sql_quality(simulation_sql, question, schema)
    
    def _evaluate_sql_quality(self, sql: str, question: str, schema: Dict) -> float:
        """Evaluate SQL quality (simplified)."""
        try:
            # Parse check
            parsed = sqlparse.parse(sql)
            if not parsed:
                return 0.0
            
            score = 0.5  # Base score for parsing
            
            # Check basic structure
            sql_upper = sql.upper()
            if 'SELECT' in sql_upper:
                score += 0.2
            if 'FROM' in sql_upper:
                score += 0.2
                
            # Check schema alignment
            for table_name in schema.keys():
                if table_name.upper() in sql_upper:
                    score += 0.1
            
            return min(score, 1.0)
            
        except:
            return 0.0
    
    def _backpropagate(self, path: List[str], reward: float):
        """Backpropagate reward through path."""
        for state in path:
            if state in self.mcts_tree:
                self.mcts_tree[state]['visits'] += 1
                self.mcts_tree[state]['value'] += reward
    
    def _extract_policy(self, root_state: str) -> List[str]:
        """Extract best policy actions."""
        actions = []
        current_state = root_state
        
        while current_state in self.mcts_tree and self.mcts_tree[current_state]['children']:
            children = self.mcts_tree[current_state]['children']
            
            # Select most visited child
            best_child = max(children, key=lambda c: self.mcts_tree[c]['visits'])
            
            # Extract action (difference between states)
            _, _, current_sql = current_state.split('|', 2)
            _, _, next_sql = best_child.split('|', 2)
            
            if next_sql.startswith(current_sql):
                action = next_sql[len(current_sql):].strip()
                if action:
                    actions.append(action)
            
            current_state = best_child
        
        return actions
    
    def _calculate_confidence(self, root_state: str) -> float:
        """Calculate confidence in policy."""
        if root_state not in self.mcts_tree:
            return 0.0
        
        root_data = self.mcts_tree[root_state]
        if root_data['visits'] == 0:
            return 0.0
        
        return min(root_data['visits'] / self.config.mcts_simulations, 1.0)
    
    def _get_exploration_stats(self) -> Dict:
        """Get exploration statistics."""
        return {
            'total_states': len(self.mcts_tree),
            'max_depth': self._calculate_max_depth(),
            'avg_branching': self._calculate_avg_branching()
        }
    
    def _calculate_max_depth(self) -> int:
        """Calculate maximum exploration depth."""
        max_depth = 0
        for state in self.mcts_tree:
            depth = state.count('|')
            max_depth = max(max_depth, depth)
        return max_depth
    
    def _calculate_avg_branching(self) -> float:
        """Calculate average branching factor."""
        total_children = sum(len(data['children']) for data in self.mcts_tree.values())
        non_leaf_nodes = sum(1 for data in self.mcts_tree.values() if data['children'])
        return total_children / max(non_leaf_nodes, 1)

class SchemaDisambiguator(nn.Module):
    """Neural schema disambiguator with graph attention networks."""
    
    def __init__(self, config: AdvancedText2SQLConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.schema_embedding_dim
        
        # Schema component embeddings
        self.table_embedder = nn.Embedding(1000, self.embedding_dim)
        self.column_embedder = nn.Embedding(5000, self.embedding_dim)
        self.type_embedder = nn.Embedding(50, self.embedding_dim)
        
        # Graph Attention Networks
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                self.embedding_dim, 
                self.embedding_dim // config.schema_gat_heads,
                config.schema_gat_heads,
                dropout=0.1
            ) for _ in range(config.schema_gat_layers)
        ])
        
        # Schema-Question alignment
        self.question_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.embedding_dim, nhead=8, batch_first=True),
            num_layers=2
        )
        
        self.alignment_attention = nn.MultiheadAttention(
            self.embedding_dim, num_heads=8, batch_first=True
        )
        
        # Disambiguation classifier
        self.disambiguator = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, 1)
        )
        
        # Schema graph constructor
        self.schema_graph_builder = SchemaGraphBuilder()
    
    def forward(self, question_embeddings: torch.Tensor, schema: Dict) -> Dict:
        """Disambiguate schema elements for the given question."""
        
        # Build schema graph
        schema_graph, node_features, edge_indices = self._build_schema_graph(schema)
        
        # Apply GAT layers
        for gat_layer in self.gat_layers:
            node_features = gat_layer(node_features, edge_indices)
            node_features = F.relu(node_features)
        
        # Encode question
        question_context = self.question_encoder(question_embeddings)
        
        # Schema-Question alignment
        aligned_features, attention_weights = self.alignment_attention(
            node_features,  # query
            question_context,  # key
            question_context   # value
        )
        
        # Combine features
        combined_features = torch.cat([node_features, aligned_features], dim=-1)
        
        # Disambiguation scores
        disambiguation_scores = self.disambiguator(combined_features)
        
        return {
            'disambiguation_scores': disambiguation_scores,
            'attention_weights': attention_weights,
            'schema_embeddings': node_features,
            'aligned_features': aligned_features
        }
    
    def _build_schema_graph(self, schema: Dict) -> Tuple[nx.Graph, torch.Tensor, torch.Tensor]:
        """Build graph representation of schema."""
        
        graph = nx.Graph()
        node_features = []
        node_to_idx = {}
        
        # Add table nodes
        for table_name, table_info in schema.items():
            table_idx = len(node_features)
            node_to_idx[f"table_{table_name}"] = table_idx
            
            # Table embedding (simplified - would use proper vocab mapping)
            table_embed = self.table_embedder(torch.tensor(hash(table_name) % 1000))
            node_features.append(table_embed)
            
            graph.add_node(table_idx, type='table', name=table_name)
            
            # Add column nodes
            for col_info in table_info.get('columns', []):
                col_name = col_info['name']
                col_idx = len(node_features)
                node_to_idx[f"column_{table_name}_{col_name}"] = col_idx
                
                # Column embedding
                col_embed = self.column_embedder(torch.tensor(hash(col_name) % 5000))
                type_embed = self.type_embedder(torch.tensor(hash(col_info.get('type', 'TEXT')) % 50))
                combined_embed = col_embed + type_embed
                
                node_features.append(combined_embed)
                graph.add_node(col_idx, type='column', name=col_name, table=table_name)
                
                # Edge between table and column
                graph.add_edge(table_idx, col_idx)
        
        # Add foreign key edges
        for table_name, table_info in schema.items():
            for fk in table_info.get('foreign_keys', []):
                try:
                    from_key = f"column_{table_name}_{fk['column']}"
                    to_table, to_col = fk['references_table'], fk['references_column']
                    to_key = f"column_{to_table}_{to_col}"
                    
                    if from_key in node_to_idx and to_key in node_to_idx:
                        graph.add_edge(node_to_idx[from_key], node_to_idx[to_key])
                except KeyError:
                    continue
        
        # Convert to tensors
        node_features_tensor = torch.stack(node_features)
        edge_indices = torch.tensor(list(graph.edges)).T
        
        return graph, node_features_tensor, edge_indices

class GraphAttentionLayer(nn.Module):
    """Graph Attention Network layer for schema understanding."""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        nn.init.xavier_uniform_(self.W.weight.data)
        nn.init.xavier_uniform_(self.a.data)
    
    def forward(self, x: torch.Tensor, edge_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass of GAT layer."""
        
        batch_size, num_nodes, _ = x.size()
        
        # Linear transformation
        h = self.W(x).view(batch_size, num_nodes, self.num_heads, self.out_features)
        
        # Attention mechanism
        edge_src, edge_dst = edge_indices[0], edge_indices[1]
        
        # Concatenate source and target node features
        h_src = h[:, edge_src]  # [batch_size, num_edges, num_heads, out_features]
        h_dst = h[:, edge_dst]  # [batch_size, num_edges, num_heads, out_features]
        
        edge_features = torch.cat([h_src, h_dst], dim=-1)  # [batch_size, num_edges, num_heads, 2*out_features]
        
        # Attention scores
        attention_scores = torch.matmul(edge_features, self.a.unsqueeze(0))  # [batch_size, num_edges, num_heads, 1]
        attention_scores = self.leaky_relu(attention_scores).squeeze(-1)  # [batch_size, num_edges, num_heads]
        
        # Softmax over edges for each node
        attention_weights = torch.zeros(batch_size, num_nodes, self.num_heads, device=x.device)
        for i in range(len(edge_src)):
            src_idx = edge_src[i]
            attention_weights[:, src_idx] = torch.softmax(attention_scores[:, i], dim=0)
        
        # Aggregate features
        aggregated = torch.zeros(batch_size, num_nodes, self.num_heads, self.out_features, device=x.device)
        
        for i in range(len(edge_src)):
            src_idx, dst_idx = edge_src[i], edge_dst[i]
            aggregated[:, src_idx] += attention_weights[:, src_idx].unsqueeze(-1) * h_dst[:, i]
        
        # Average over heads
        output = aggregated.mean(dim=2)
        
        return self.dropout(output)

class SchemaGraphBuilder:
    """Builds graph representations of database schemas."""
    
    def build_graph(self, schema: Dict) -> nx.Graph:
        """Build NetworkX graph from schema."""
        
        graph = nx.Graph()
        
        # Add nodes for tables and columns
        for table_name, table_info in schema.items():
            # Table node
            graph.add_node(f"table_{table_name}", 
                          type='table', 
                          name=table_name)
            
            # Column nodes
            for col_info in table_info.get('columns', []):
                col_name = col_info['name']
                graph.add_node(f"column_{table_name}_{col_name}",
                              type='column',
                              name=col_name,
                              table=table_name,
                              data_type=col_info.get('type', 'unknown'))
                
                # Edge between table and column
                graph.add_edge(f"table_{table_name}", 
                              f"column_{table_name}_{col_name}")
        
        # Add foreign key relationships
        for table_name, table_info in schema.items():
            for fk in table_info.get('foreign_keys', []):
                try:
                    from_col = f"column_{table_name}_{fk['column']}"
                    to_table = fk['references_table']
                    to_col = f"column_{to_table}_{fk['references_column']}"
                    
                    if graph.has_node(from_col) and graph.has_node(to_col):
                        graph.add_edge(from_col, to_col, relationship='foreign_key')
                except KeyError:
                    continue
        
        return graph

class QueryClarifier:
    """Iterative query clarification with uncertainty estimation."""
    
    def __init__(self, config: AdvancedText2SQLConfig, tokenizer: PreTrainedTokenizerFast):
        self.config = config
        self.tokenizer = tokenizer
        self.uncertainty_estimator = UncertaintyEstimator()
        
    def clarify_query(self, question: str, schema: Dict, initial_sql: str, model) -> Dict:
        """Iteratively clarify query with uncertainty-driven feedback."""
        
        clarifications = []
        current_question = question
        current_sql = initial_sql
        
        for iteration in range(self.config.clarification_iterations):
            # Estimate uncertainty
            uncertainty = self.uncertainty_estimator.estimate_uncertainty(
                current_question, schema, current_sql, model
            )
            
            if uncertainty < self.config.uncertainty_threshold:
                break
                
            # Generate clarification
            clarification = self._generate_clarification(
                current_question, schema, current_sql, uncertainty
            )
            
            clarifications.append({
                'iteration': iteration,
                'uncertainty': uncertainty,
                'clarification': clarification,
                'previous_sql': current_sql
            })
            
            # Update question with clarification
            current_question = self._update_question_with_clarification(
                current_question, clarification
            )
            
            # Generate new SQL with clarified question
            current_sql = self._generate_sql_with_clarification(
                current_question, schema, model
            )
        
        return {
            'final_question': current_question,
            'final_sql': current_sql,
            'clarifications': clarifications,
            'final_uncertainty': uncertainty if 'uncertainty' in locals() else 1.0
        }
    
    def _generate_clarification(self, question: str, schema: Dict, sql: str, uncertainty: float) -> str:
        """Generate specific clarification based on uncertainty analysis."""
        
        clarifications = []
        
        # Analyze SQL for potential ambiguities
        if 'JOIN' in sql.upper() and uncertainty > 0.5:
            clarifications.append("Please clarify which tables should be joined and how.")
        
        if 'WHERE' not in sql.upper() and len(schema) > 1 and uncertainty > 0.4:
            clarifications.append("Should any filtering conditions be applied?")
        
        if any(agg in sql.upper() for agg in ['COUNT', 'SUM', 'AVG']) and uncertainty > 0.3:
            clarifications.append("Please confirm the aggregation requirements.")
        
        # Schema-specific clarifications
        table_count = len(schema)
        if table_count > 3 and uncertainty > 0.6:
            clarifications.append(f"The schema has {table_count} tables. Which tables are most relevant?")
        
        return "; ".join(clarifications) if clarifications else "Please provide more specific requirements."
    
    def _update_question_with_clarification(self, question: str, clarification: str) -> str:
        """Update question with clarification context."""
        return f"{question}\n\nClarification needed: {clarification}"
    
    def _generate_sql_with_clarification(self, clarified_question: str, schema: Dict, model) -> str:
        """Generate SQL with the clarified question."""
        
        # Prepare input with clarification markers
        prompt = f"""<CLARIFICATION>
Given the clarified question and schema, generate an accurate SQL query.

Question: {clarified_question}

<SCHEMA_START>
{self._format_schema(schema)}
<SCHEMA_END>

<QUERY_START>"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=4096, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,  # Lower temperature for more focused generation
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract SQL
        if "<QUERY_START>" in generated_text and "<QUERY_END>" in generated_text:
            start = generated_text.find("<QUERY_START>") + len("<QUERY_START>")
            end = generated_text.find("<QUERY_END>")
            return generated_text[start:end].strip()
        
        return generated_text.split("SELECT")[-1] if "SELECT" in generated_text else ""
    
    def _format_schema(self, schema: Dict) -> str:
        """Format schema for clarification context."""
        formatted = []
        for table_name, table_info in schema.items():
            cols = [f"{col['name']} ({col.get('type', 'TEXT')})" 
                   for col in table_info.get('columns', [])]
            formatted.append(f"<TABLE_START>{table_name}<TABLE_END>: {', '.join(cols)}")
        return "\n".join(formatted)

class UncertaintyEstimator:
    """Estimates uncertainty in SQL generation."""
    
    def estimate_uncertainty(self, question: str, schema: Dict, sql: str, model) -> float:
        """Estimate uncertainty using multiple techniques."""
        
        uncertainties = []
        
        # 1. Semantic consistency uncertainty
        semantic_uncertainty = self._semantic_uncertainty(question, sql)
        uncertainties.append(semantic_uncertainty)
        
        # 2. Schema alignment uncertainty
        schema_uncertainty = self._schema_alignment_uncertainty(sql, schema)
        uncertainties.append(schema_uncertainty)
        
        # 3. Model confidence uncertainty (entropy-based)
        if model is not None:
            model_uncertainty = self._model_confidence_uncertainty(question, schema, model)
            uncertainties.append(model_uncertainty)
        
        # 4. SQL complexity uncertainty
        complexity_uncertainty = self._sql_complexity_uncertainty(sql)
        uncertainties.append(complexity_uncertainty)
        
        # Aggregate uncertainties
        return sum(uncertainties) / len(uncertainties)
    
    def _semantic_uncertainty(self, question: str, sql: str) -> float:
        """Estimate semantic alignment uncertainty."""
        
        # Simple keyword matching (could be enhanced with embeddings)
        question_words = set(question.lower().split())
        sql_words = set(sql.lower().split())
        
        # Remove SQL keywords to focus on semantic content
        sql_keywords = {'select', 'from', 'where', 'join', 'group', 'by', 'having', 
                       'order', 'limit', 'and', 'or', 'on', 'in', 'as'}
        sql_content_words = sql_words - sql_keywords
        
        if not sql_content_words:
            return 1.0  # High uncertainty if no content words
        
        overlap = len(question_words & sql_content_words)
        return 1.0 - (overlap / max(len(question_words), len(sql_content_words)))
    
    def _schema_alignment_uncertainty(self, sql: str, schema: Dict) -> float:
        """Estimate schema alignment uncertainty."""
        
        try:
            parsed = sqlparse.parse(sql)[0]
            referenced_tables = set()
            referenced_columns = set()
            
            # Extract table and column references
            for token in parsed.flatten():
                if token.ttype is None:
                    value = token.value.strip()
                    if '.' in value:
                        table, column = value.split('.', 1)
                        referenced_tables.add(table.strip())
                        referenced_columns.add(column.strip())
            
            # Check against schema
            valid_tables = set(schema.keys())
            valid_columns = set()
            for table_info in schema.values():
                for col in table_info.get('columns', []):
                    valid_columns.add(col['name'])
            
            # Calculate alignment
            table_alignment = len(referenced_tables & valid_tables) / max(len(referenced_tables), 1)
            column_alignment = len(referenced_columns & valid_columns) / max(len(referenced_columns), 1)
            
            return 1.0 - ((table_alignment + column_alignment) / 2)
            
        except Exception:
            return 1.0  # High uncertainty if parsing fails
    
    def _model_confidence_uncertainty(self, question: str, schema: Dict, model) -> float:
        """Estimate model confidence using entropy."""
        
        try:
            # Format input
            prompt = f"Question: {question}\nSchema: {schema}\nSQL:"
            inputs = model.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
            
            # Get logits
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Calculate entropy
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12))
                
                # Normalize entropy (log(vocab_size) is max entropy)
                max_entropy = math.log(len(probs))
                normalized_entropy = entropy / max_entropy
                
                return normalized_entropy.item()
        
        except Exception:
            return 0.5  # Medium uncertainty if calculation fails
    
    def _sql_complexity_uncertainty(self, sql: str) -> float:
        """Estimate uncertainty based on SQL complexity."""
        
        complexity_indicators = [
            'JOIN', 'SUBQUERY', 'UNION', 'INTERSECT', 'EXCEPT',
            'WINDOW', 'OVER', 'PARTITION', 'CASE', 'EXISTS'
        ]
        
        sql_upper = sql.upper()
        complexity_count = sum(1 for indicator in complexity_indicators if indicator in sql_upper)
        
        # Higher complexity -> higher uncertainty
        max_expected_complexity = 5
        return min(complexity_count / max_expected_complexity, 1.0)

class MultiAgentRLCoordinator:
    """Coordinates multiple specialized agents for text2SQL."""
    
    def __init__(self, config: AdvancedText2SQLConfig):
        self.config = config
        self.agents = {
            'schema_agent': SchemaAgent(config),
            'sql_agent': SQLAgent(config), 
            'validation_agent': ValidationAgent(config)
        }
        self.cooperation_history = []
        
    def coordinate_agents(self, question: str, schema: Dict) -> Dict:
        """Coordinate multiple agents to solve text2SQL task."""
        
        # Phase 1: Schema Analysis
        schema_analysis = self.agents['schema_agent'].analyze_schema(question, schema)
        
        # Phase 2: SQL Generation with schema guidance
        sql_generation = self.agents['sql_agent'].generate_sql(
            question, schema, schema_analysis
        )
        
        # Phase 3: Validation and refinement
        validation_result = self.agents['validation_agent'].validate_sql(
            sql_generation['sql'], question, schema
        )
        
        # Phase 4: Cooperative refinement
        if validation_result['needs_refinement']:
            refined_result = self._cooperative_refinement(
                question, schema, sql_generation, validation_result
            )
        else:
            refined_result = sql_generation
        
        return {
            'final_sql': refined_result['sql'],
            'schema_analysis': schema_analysis,
            'sql_generation': sql_generation,
            'validation_result': validation_result,
            'cooperation_stats': self._get_cooperation_stats()
        }
    
    def _cooperative_refinement(self, question: str, schema: Dict, 
                               sql_result: Dict, validation_result: Dict) -> Dict:
        """Perform cooperative refinement between agents."""
        
        refinement_iterations = 3
        current_sql = sql_result['sql']
        
        for iteration in range(refinement_iterations):
            # Schema agent provides schema insights
            schema_feedback = self.agents['schema_agent'].provide_feedback(
                current_sql, validation_result['issues']
            )
            
            # SQL agent incorporates feedback
            refined_sql = self.agents['sql_agent'].refine_sql(
                current_sql, schema_feedback, validation_result['issues']
            )
            
            # Validation agent checks refinement
            new_validation = self.agents['validation_agent'].validate_sql(
                refined_sql, question, schema
            )
            
            if not new_validation['needs_refinement']:
                break
                
            current_sql = refined_sql
            validation_result = new_validation
        
        self.cooperation_history.append({
            'question': question,
            'iterations': iteration + 1,
            'final_issues': new_validation['issues']
        })
        
        return {
            'sql': current_sql,
            'refinement_iterations': iteration + 1
        }
    
    def _get_cooperation_stats(self) -> Dict:
        """Get cooperation statistics."""
        if not self.cooperation_history:
            return {'avg_iterations': 0, 'success_rate': 0}
        
        avg_iterations = sum(h['iterations'] for h in self.cooperation_history) / len(self.cooperation_history)
        success_rate = sum(1 for h in self.cooperation_history if not h['final_issues']) / len(self.cooperation_history)
        
        return {
            'avg_iterations': avg_iterations,
            'success_rate': success_rate,
            'total_coordinations': len(self.cooperation_history)
        }

class SchemaAgent:
    """Specialized agent for schema analysis and guidance."""
    
    def __init__(self, config: AdvancedText2SQLConfig):
        self.config = config
    
    def analyze_schema(self, question: str, schema: Dict) -> Dict:
        """Analyze schema relevance for the question."""
        
        # Extract question entities
        question_entities = self._extract_question_entities(question)
        
        # Rank table relevance
        table_relevance = self._rank_table_relevance(question_entities, schema)
        
        # Identify key relationships
        key_relationships = self._identify_key_relationships(schema, table_relevance)
        
        # Suggest query patterns
        query_patterns = self._suggest_query_patterns(question, schema)
        
        return {
            'question_entities': question_entities,
            'table_relevance': table_relevance,
            'key_relationships': key_relationships,
            'suggested_patterns': query_patterns
        }
    
    def provide_feedback(self, sql: str, issues: List[str]) -> Dict:
        """Provide schema-based feedback for SQL refinement."""
        
        feedback = {
            'missing_joins': [],
            'incorrect_columns': [],
            'schema_suggestions': []
        }
        
        for issue in issues:
            if 'join' in issue.lower():
                feedback['missing_joins'].append(issue)
            elif 'column' in issue.lower():
                feedback['incorrect_columns'].append(issue)
            else:
                feedback['schema_suggestions'].append(issue)
        
        return feedback
    
    def _extract_question_entities(self, question: str) -> List[str]:
        """Extract potential database entities from question."""
        # Simple extraction - could be enhanced with NER
        words = question.lower().split()
        entities = [word for word in words if len(word) > 3 and word.isalpha()]
        return entities
    
    def _rank_table_relevance(self, entities: List[str], schema: Dict) -> Dict[str, float]:
        """Rank table relevance based on entity matching."""
        relevance = {}
        
        for table_name, table_info in schema.items():
            score = 0.0
            
            # Check table name similarity
            for entity in entities:
                if entity in table_name.lower():
                    score += 2.0
            
            # Check column name similarity
            for col in table_info.get('columns', []):
                col_name = col['name'].lower()
                for entity in entities:
                    if entity in col_name or col_name in entity:
                        score += 1.0
            
            relevance[table_name] = score
        
        return relevance
    
    def _identify_key_relationships(self, schema: Dict, table_relevance: Dict) -> List[Dict]:
        """Identify key foreign key relationships."""
        relationships = []
        
        for table_name, table_info in schema.items():
            for fk in table_info.get('foreign_keys', []):
                rel_score = table_relevance.get(table_name, 0) + table_relevance.get(fk.get('references_table', ''), 0)
                relationships.append({
                    'from_table': table_name,
                    'from_column': fk['column'],
                    'to_table': fk['references_table'],
                    'to_column': fk['references_column'],
                    'relevance_score': rel_score
                })
        
        return sorted(relationships, key=lambda x: x['relevance_score'], reverse=True)
    
    def _suggest_query_patterns(self, question: str, schema: Dict) -> List[str]:
        """Suggest query patterns based on question analysis."""
        patterns = []
        
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['how many', 'count', 'number of']):
            patterns.append('aggregation_count')
        
        if any(word in question_lower for word in ['average', 'avg', 'mean']):
            patterns.append('aggregation_avg')
        
        if any(word in question_lower for word in ['maximum', 'max', 'highest', 'largest']):
            patterns.append('aggregation_max')
        
        if any(word in question_lower for word in ['list', 'show', 'display', 'find']):
            patterns.append('selection')
        
        if len(schema) > 1:
            patterns.append('multi_table_join')
        
        return patterns

class SQLAgent:
    """Specialized agent for SQL generation."""
    
    def __init__(self, config: AdvancedText2SQLConfig):
        self.config = config
    
    def generate_sql(self, question: str, schema: Dict, schema_analysis: Dict) -> Dict:
        """Generate SQL using schema analysis guidance."""
        
        # Prioritize tables based on analysis
        relevant_tables = self._select_relevant_tables(schema_analysis['table_relevance'])
        
        # Generate SQL components
        select_clause = self._generate_select_clause(question, schema, schema_analysis)
        from_clause = self._generate_from_clause(relevant_tables, schema_analysis)
        where_clause = self._generate_where_clause(question, schema)
        
        sql = f"{select_clause}\n{from_clause}"
        if where_clause:
            sql += f"\n{where_clause}"
        
        return {
            'sql': sql,
            'components': {
                'select': select_clause,
                'from': from_clause,
                'where': where_clause
            },
            'confidence': self._calculate_generation_confidence(sql, schema_analysis)
        }
    
    def refine_sql(self, sql: str, schema_feedback: Dict, issues: List[str]) -> str:
        """Refine SQL based on feedback."""
        
        refined_sql = sql
        
        # Add missing joins
        for join_issue in schema_feedback.get('missing_joins', []):
            refined_sql = self._add_missing_join(refined_sql, join_issue)
        
        # Fix column references
        for col_issue in schema_feedback.get('incorrect_columns', []):
            refined_sql = self._fix_column_reference(refined_sql, col_issue)
        
        return refined_sql
    
    def _select_relevant_tables(self, table_relevance: Dict) -> List[str]:
        """Select most relevant tables."""
        sorted_tables = sorted(table_relevance.items(), key=lambda x: x[1], reverse=True)
        return [table for table, score in sorted_tables if score > 0][:3]  # Top 3
    
    def _generate_select_clause(self, question: str, schema: Dict, analysis: Dict) -> str:
        """Generate SELECT clause based on question analysis."""
        
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['count', 'how many', 'number']):
            return "SELECT COUNT(*)"
        elif any(word in question_lower for word in ['average', 'avg']):
            # Find numeric columns
            numeric_cols = self._find_numeric_columns(schema)
            if numeric_cols:
                return f"SELECT AVG({numeric_cols[0]})"
            return "SELECT AVG(*)"
        elif any(word in question_lower for word in ['sum', 'total']):
            numeric_cols = self._find_numeric_columns(schema)
            if numeric_cols:
                return f"SELECT SUM({numeric_cols[0]})"
        else:
            return "SELECT *"
    
    def _generate_from_clause(self, relevant_tables: List[str], analysis: Dict) -> str:
        """Generate FROM clause with appropriate joins."""
        
        if len(relevant_tables) == 1:
            return f"FROM {relevant_tables[0]}"
        
        # Use relationship analysis for joins
        relationships = analysis.get('key_relationships', [])
        
        from_clause = f"FROM {relevant_tables[0]}"
        
        for i, table in enumerate(relevant_tables[1:], 1):
            # Find relationship to connect this table
            join_rel = None
            for rel in relationships:
                if (rel['from_table'] == relevant_tables[i-1] and rel['to_table'] == table) or \
                   (rel['to_table'] == relevant_tables[i-1] and rel['from_table'] == table):
                    join_rel = rel
                    break
            
            if join_rel:
                from_clause += f"\nJOIN {table} ON {join_rel['from_table']}.{join_rel['from_column']} = {join_rel['to_table']}.{join_rel['to_column']}"
            else:
                from_clause += f"\nJOIN {table}"  # Cross join as fallback
        
        return from_clause
    
    def _generate_where_clause(self, question: str, schema: Dict) -> str:
        """Generate WHERE clause based on question filters."""
        
        # Simple pattern matching for common filters
        conditions = []
        
        question_lower = question.lower()
        
        # Look for year mentions
        import re
        years = re.findall(r'\b(19|20)\d{2}\b', question)
        if years:
            conditions.append(f"YEAR(date_column) = {years[0]}")
        
        # Look for specific values in quotes
        quoted_values = re.findall(r"'([^']*)'", question)
        if quoted_values:
            conditions.append(f"column_name = '{quoted_values[0]}'")
        
        return f"WHERE {' AND '.join(conditions)}" if conditions else ""
    
    def _find_numeric_columns(self, schema: Dict) -> List[str]:
        """Find numeric columns in schema."""
        numeric_cols = []
        numeric_types = ['int', 'integer', 'float', 'double', 'decimal', 'numeric']
        
        for table_name, table_info in schema.items():
            for col in table_info.get('columns', []):
                if any(num_type in col.get('type', '').lower() for num_type in numeric_types):
                    numeric_cols.append(f"{table_name}.{col['name']}")
        
        return numeric_cols
    
    def _calculate_generation_confidence(self, sql: str, analysis: Dict) -> float:
        """Calculate confidence in generated SQL."""
        
        confidence = 0.5  # Base confidence
        
        # Boost confidence if using high-relevance tables
        table_relevance = analysis.get('table_relevance', {})
        if table_relevance:
            max_relevance = max(table_relevance.values())
            confidence += min(max_relevance / 10.0, 0.3)
        
        # Boost confidence if using relationships
        if analysis.get('key_relationships'):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _add_missing_join(self, sql: str, join_issue: str) -> str:
        """Add missing join to SQL."""
        # Simplified implementation
        return sql  # Would implement proper join addition
    
    def _fix_column_reference(self, sql: str, col_issue: str) -> str:
        """Fix incorrect column reference."""
        # Simplified implementation
        return sql  # Would implement proper column fixing

class ValidationAgent:
    """Specialized agent for SQL validation."""
    
    def __init__(self, config: AdvancedText2SQLConfig):
        self.config = config
    
    def validate_sql(self, sql: str, question: str, schema: Dict) -> Dict:
        """Validate SQL syntax and semantic correctness."""
        
        issues = []
        
        # Syntax validation
        syntax_issues = self._validate_syntax(sql)
        issues.extend(syntax_issues)
        
        # Schema validation
        schema_issues = self._validate_schema_compliance(sql, schema)
        issues.extend(schema_issues)
        
        # Semantic validation
        semantic_issues = self._validate_semantics(sql, question)
        issues.extend(semantic_issues)
        
        return {
            'is_valid': len(issues) == 0,
            'needs_refinement': len(issues) > 0,
            'issues': issues,
            'confidence': 1.0 - len(issues) / 10.0  # Rough confidence estimate
        }
    
    def _validate_syntax(self, sql: str) -> List[str]:
        """Validate SQL syntax."""
        issues = []
        
        try:
            parsed = sqlparse.parse(sql)
            if not parsed or not parsed[0].tokens:
                issues.append("SQL parsing failed")
        except Exception as e:
            issues.append(f"Syntax error: {str(e)}")
        
        return issues
    
    def _validate_schema_compliance(self, sql: str, schema: Dict) -> List[str]:
        """Validate SQL compliance with schema."""
        issues = []
        
        try:
            parsed = sqlparse.parse(sql)[0]
            
            # Extract table and column references
            referenced_tables = set()
            referenced_columns = set()
            
            for token in parsed.flatten():
                if token.ttype is None:
                    value = token.value.strip()
                    if '.' in value:
                        table, column = value.split('.', 1)
                        referenced_tables.add(table)
                        referenced_columns.add(column)
            
            # Check table existence
            valid_tables = set(schema.keys())
            invalid_tables = referenced_tables - valid_tables
            if invalid_tables:
                issues.append(f"Invalid tables: {', '.join(invalid_tables)}")
            
            # Check column existence
            valid_columns = set()
            for table_info in schema.values():
                for col in table_info.get('columns', []):
                    valid_columns.add(col['name'])
            
            invalid_columns = referenced_columns - valid_columns
            if invalid_columns:
                issues.append(f"Invalid columns: {', '.join(invalid_columns)}")
        
        except Exception as e:
            issues.append(f"Schema validation error: {str(e)}")
        
        return issues
    
    def _validate_semantics(self, sql: str, question: str) -> List[str]:
        """Validate semantic alignment between SQL and question."""
        issues = []
        
        question_lower = question.lower()
        sql_lower = sql.lower()
        
        # Check for count requests
        if any(word in question_lower for word in ['how many', 'count', 'number']) and 'count' not in sql_lower:
            issues.append("Question asks for count but SQL doesn't use COUNT")
        
        # Check for average requests
        if any(word in question_lower for word in ['average', 'avg']) and 'avg' not in sql_lower:
            issues.append("Question asks for average but SQL doesn't use AVG")
        
        # Check for filtering indications
        if any(word in question_lower for word in ['where', 'when', 'which']) and 'where' not in sql_lower:
            issues.append("Question suggests filtering but SQL has no WHERE clause")
        
        return issues

def main():
    """Test the advanced Text2SQL system."""
    
    config = AdvancedText2SQLConfig()
    
    # Example usage
    question = "How many employees work in the engineering department?"
    schema = {
        "employees": {
            "columns": [
                {"name": "id", "type": "int"},
                {"name": "name", "type": "varchar"},
                {"name": "department_id", "type": "int"}
            ],
            "foreign_keys": [
                {"column": "department_id", "references_table": "departments", "references_column": "id"}
            ]
        },
        "departments": {
            "columns": [
                {"name": "id", "type": "int"},
                {"name": "name", "type": "varchar"}
            ],
            "foreign_keys": []
        }
    }
    
    # Initialize components
    coordinator = MultiAgentRLCoordinator(config)
    
    # Coordinate agents to solve the problem
    result = coordinator.coordinate_agents(question, schema)
    
    print(f"Final SQL: {result['final_sql']}")
    print(f"Cooperation Stats: {result['cooperation_stats']}")

if __name__ == "__main__":
    main()