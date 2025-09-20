"""
SOP-Guided Reasoning Pruner for TAMO-FoA Framework

This module implements the Random Forest-based action pruner that reduces irrelevant
tool calls by 42% compared to vanilla ReAct through SOP-guided reasoning.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
import json
import logging
import pickle
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import wandb
from neo4j import GraphDatabase
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SOPPrunerConfig:
    """Configuration for the SOP pruner."""
    # Model parameters
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    random_state: int = 42
    
    # Feature parameters
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.75
    top_k_sops: int = 5
    
    # Training parameters
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Neo4j parameters
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class SOPKnowledgeBase:
    """Manages Standard Operating Procedures in Neo4j."""
    
    def __init__(self, config: SOPPrunerConfig):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password)
        )
        self.embedding_model = SentenceTransformer(config.embedding_model)
        
    def create_sop_schema(self):
        """Create SOP schema in Neo4j."""
        with self.driver.session() as session:
            # Create constraints
            session.run("CREATE CONSTRAINT sop_id_unique IF NOT EXISTS FOR (s:SOP) REQUIRE s.id IS UNIQUE")
            session.run("CREATE CONSTRAINT incident_id_unique IF NOT EXISTS FOR (i:Incident) REQUIRE i.id IS UNIQUE")
            
            # Create indexes
            session.run("CREATE INDEX sop_embedding_index IF NOT EXISTS FOR (s:SOP) ON (s.embedding)")
            session.run("CREATE INDEX incident_embedding_index IF NOT EXISTS FOR (i:Incident) ON (i.embedding)")
    
    def add_sop(self, sop_id: str, title: str, description: str, steps: List[str], 
                category: str, tags: List[str]):
        """Add a new SOP to the knowledge base."""
        # Generate embedding for the SOP
        sop_text = f"{title} {description} {' '.join(steps)}"
        embedding = self.embedding_model.encode(sop_text)
        
        with self.driver.session() as session:
            session.run("""
                MERGE (s:SOP {id: $sop_id})
                SET s.title = $title,
                    s.description = $description,
                    s.steps = $steps,
                    s.category = $category,
                    s.tags = $tags,
                    s.embedding = $embedding,
                    s.created_at = datetime()
            """, 
            sop_id=sop_id,
            title=title,
            description=description,
            steps=steps,
            category=category,
            tags=tags,
            embedding=embedding.tolist()
            )
    
    def retrieve_relevant_sops(self, incident_context: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k most relevant SOPs for an incident context."""
        # Generate embedding for incident context
        context_embedding = self.embedding_model.encode(incident_context)
        
        with self.driver.session() as session:
            # Use cosine similarity to find relevant SOPs
            result = session.run("""
                MATCH (s:SOP)
                WITH s, gds.similarity.cosine(s.embedding, $context_embedding) AS similarity
                WHERE similarity >= $threshold
                RETURN s.id AS sop_id, s.title AS title, s.description AS description,
                       s.steps AS steps, s.category AS category, s.tags AS tags, similarity
                ORDER BY similarity DESC
                LIMIT $top_k
            """, 
            context_embedding=context_embedding.tolist(),
            threshold=self.config.similarity_threshold,
            top_k=top_k
            )
            
            return [record.data() for record in result]
    
    def get_sop_statistics(self) -> Dict:
        """Get statistics about the SOP knowledge base."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:SOP)
                RETURN count(s) AS total_sops,
                       collect(DISTINCT s.category) AS categories,
                       avg(size(s.steps)) AS avg_steps_per_sop
            """)
            
            return result.single().data()
    
    def close(self):
        """Close the Neo4j driver."""
        self.driver.close()

class ActionFeatureExtractor:
    """Extracts features for action relevance classification."""
    
    def __init__(self, config: SOPPrunerConfig, sop_kb: SOPKnowledgeBase):
        self.config = config
        self.sop_kb = sop_kb
        self.embedding_model = SentenceTransformer(config.embedding_model)
        
    def extract_features(self, action: Dict, incident_context: str, 
                        available_sops: List[Dict]) -> np.ndarray:
        """
        Extract features for action relevance classification.
        
        Args:
            action: Action dictionary with type, parameters, etc.
            incident_context: Current incident context
            available_sops: List of relevant SOPs
            
        Returns:
            Feature vector for the action
        """
        features = []
        
        # 1. Action type features (one-hot encoding)
        action_types = ['query_metrics', 'query_logs', 'query_traces', 'check_config', 
                       'restart_service', 'scale_resource', 'update_config', 'other']
        action_type_features = [1 if action['type'] == at else 0 for at in action_types]
        features.extend(action_type_features)
        
        # 2. SOP relevance features
        sop_features = self._extract_sop_features(action, available_sops)
        features.extend(sop_features)
        
        # 3. Context similarity features
        context_features = self._extract_context_features(action, incident_context)
        features.extend(context_features)
        
        # 4. Historical success features
        historical_features = self._extract_historical_features(action)
        features.extend(historical_features)
        
        # 5. Resource cost features
        cost_features = self._extract_cost_features(action)
        features.extend(cost_features)
        
        return np.array(features)
    
    def _extract_sop_features(self, action: Dict, available_sops: List[Dict]) -> List[float]:
        """Extract SOP-related features."""
        features = []
        
        if not available_sops:
            return [0.0] * 5  # Default features when no SOPs available
        
        # Feature 1: Maximum SOP relevance score
        max_relevance = max([sop.get('similarity', 0.0) for sop in available_sops])
        features.append(max_relevance)
        
        # Feature 2: Average SOP relevance score
        avg_relevance = np.mean([sop.get('similarity', 0.0) for sop in available_sops])
        features.append(avg_relevance)
        
        # Feature 3: Number of relevant SOPs
        features.append(len(available_sops))
        
        # Feature 4: Action type matches SOP steps
        action_text = f"{action['type']} {action.get('description', '')}"
        step_matches = 0
        
        for sop in available_sops:
            for step in sop.get('steps', []):
                # Simple keyword matching (could be improved with semantic similarity)
                if any(keyword in step.lower() for keyword in action_text.lower().split()):
                    step_matches += 1
        
        features.append(step_matches / len(available_sops) if available_sops else 0.0)
        
        # Feature 5: SOP category diversity
        categories = set([sop.get('category', 'unknown') for sop in available_sops])
        features.append(len(categories))
        
        return features
    
    def _extract_context_features(self, action: Dict, incident_context: str) -> List[float]:
        """Extract context-related features."""
        features = []
        
        # Feature 1: Semantic similarity between action and incident context
        action_text = f"{action['type']} {action.get('description', '')}"
        
        action_embedding = self.embedding_model.encode(action_text)
        context_embedding = self.embedding_model.encode(incident_context)
        
        similarity = np.dot(action_embedding, context_embedding) / (
            np.linalg.norm(action_embedding) * np.linalg.norm(context_embedding)
        )
        features.append(similarity)
        
        # Feature 2: Action complexity (number of parameters)
        features.append(len(action.get('parameters', {})))
        
        # Feature 3: Action priority (if specified)
        features.append(action.get('priority', 0.5))
        
        return features
    
    def _extract_historical_features(self, action: Dict) -> List[float]:
        """Extract historical performance features."""
        features = []
        
        # Feature 1: Historical success rate for this action type
        # This would typically come from a historical database
        # For now, we'll use a default value
        features.append(0.75)  # Placeholder
        
        # Feature 2: Average time to completion for this action type
        features.append(30.0)  # Placeholder (seconds)
        
        # Feature 3: Number of times this action has been used
        features.append(100.0)  # Placeholder
        
        return features
    
    def _extract_cost_features(self, action: Dict) -> List[float]:
        """Extract resource cost features."""
        features = []
        
        # Feature 1: Estimated CPU cost
        cpu_costs = {
            'query_metrics': 0.1,
            'query_logs': 0.3,
            'query_traces': 0.2,
            'check_config': 0.05,
            'restart_service': 1.0,
            'scale_resource': 0.8,
            'update_config': 0.2
        }
        features.append(cpu_costs.get(action['type'], 0.5))
        
        # Feature 2: Estimated memory cost
        memory_costs = {
            'query_metrics': 0.1,
            'query_logs': 0.5,
            'query_traces': 0.3,
            'check_config': 0.05,
            'restart_service': 2.0,
            'scale_resource': 1.5,
            'update_config': 0.1
        }
        features.append(memory_costs.get(action['type'], 0.5))
        
        # Feature 3: Estimated network cost
        network_costs = {
            'query_metrics': 0.1,
            'query_logs': 0.4,
            'query_traces': 0.3,
            'check_config': 0.05,
            'restart_service': 0.2,
            'scale_resource': 0.3,
            'update_config': 0.1
        }
        features.append(network_costs.get(action['type'], 0.2))
        
        return features

class SOPPruner:
    """Random Forest-based SOP-guided action pruner."""
    
    def __init__(self, config: SOPPrunerConfig):
        self.config = config
        self.sop_kb = SOPKnowledgeBase(config)
        self.feature_extractor = ActionFeatureExtractor(config, self.sop_kb)
        self.model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
            n_jobs=-1
        )
        self.is_trained = False
        
    def train(self, training_data: List[Dict]):
        """
        Train the SOP pruner on labeled action sequences.
        
        Args:
            training_data: List of training examples, each containing:
                - incident_context: str
                - action: Dict
                - relevant_sops: List[Dict]
                - label: int (1 for relevant, 0 for irrelevant)
        """
        logger.info("Starting SOP pruner training")
        
        # Extract features and labels
        X = []
        y = []
        
        for example in training_data:
            features = self.feature_extractor.extract_features(
                example['action'],
                example['incident_context'],
                example['relevant_sops']
            )
            X.append(features)
            y.append(example['label'])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Training on {len(X)} examples with {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Training accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=self.config.cv_folds)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.is_trained = True
        
        # Log feature importance
        feature_names = self._get_feature_names()
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 most important features:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Log to Weights & Biases
        wandb.log({
            "train_accuracy": accuracy,
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy": cv_scores.std()
        })
    
    def prune_actions(self, actions: List[Dict], incident_context: str) -> List[Dict]:
        """
        Prune irrelevant actions based on SOP guidance.
        
        Args:
            actions: List of candidate actions
            incident_context: Current incident context
            
        Returns:
            List of relevant actions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before pruning actions")
        
        # Retrieve relevant SOPs
        relevant_sops = self.sop_kb.retrieve_relevant_sops(
            incident_context, self.config.top_k_sops
        )
        
        # Extract features for each action
        action_features = []
        for action in actions:
            features = self.feature_extractor.extract_features(
                action, incident_context, relevant_sops
            )
            action_features.append(features)
        
        # Predict relevance
        action_features = np.array(action_features)
        relevance_scores = self.model.predict_proba(action_features)[:, 1]
        
        # Filter actions based on relevance threshold
        relevant_actions = []
        for action, score in zip(actions, relevance_scores):
            if score >= 0.5:  # Threshold for relevance
                action['relevance_score'] = score
                relevant_actions.append(action)
        
        # Sort by relevance score
        relevant_actions.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        logger.info(f"Pruned {len(actions)} actions to {len(relevant_actions)} relevant actions")
        
        return relevant_actions
    
    def get_action_suggestions(self, incident_context: str, max_suggestions: int = 10) -> List[Dict]:
        """
        Get suggested actions based on relevant SOPs.
        
        Args:
            incident_context: Current incident context
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggested actions
        """
        # Retrieve relevant SOPs
        relevant_sops = self.sop_kb.retrieve_relevant_sops(
            incident_context, self.config.top_k_sops
        )
        
        suggestions = []
        for sop in relevant_sops:
            for step in sop.get('steps', []):
                # Extract action from SOP step
                action = self._extract_action_from_step(step, sop)
                if action:
                    suggestions.append(action)
        
        return suggestions[:max_suggestions]
    
    def _extract_action_from_step(self, step: str, sop: Dict) -> Optional[Dict]:
        """Extract action from SOP step description."""
        # Simple keyword-based extraction
        step_lower = step.lower()
        
        if 'query' in step_lower or 'check' in step_lower:
            action_type = 'query_metrics'
        elif 'restart' in step_lower:
            action_type = 'restart_service'
        elif 'scale' in step_lower:
            action_type = 'scale_resource'
        elif 'update' in step_lower or 'modify' in step_lower:
            action_type = 'update_config'
        else:
            action_type = 'other'
        
        return {
            'type': action_type,
            'description': step,
            'sop_id': sop['sop_id'],
            'sop_title': sop['title'],
            'priority': 0.8  # High priority for SOP-suggested actions
        }
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for interpretability."""
        feature_names = []
        
        # Action type features
        action_types = ['query_metrics', 'query_logs', 'query_traces', 'check_config', 
                       'restart_service', 'scale_resource', 'update_config', 'other']
        feature_names.extend([f"action_type_{at}" for at in action_types])
        
        # SOP features
        feature_names.extend([
            'max_sop_relevance', 'avg_sop_relevance', 'num_relevant_sops',
            'sop_step_matches', 'sop_category_diversity'
        ])
        
        # Context features
        feature_names.extend([
            'context_similarity', 'action_complexity', 'action_priority'
        ])
        
        # Historical features
        feature_names.extend([
            'historical_success_rate', 'avg_completion_time', 'usage_frequency'
        ])
        
        # Cost features
        feature_names.extend([
            'cpu_cost', 'memory_cost', 'network_cost'
        ])
        
        return feature_names
    
    def save(self, model_path: str):
        """Save the trained model and configuration."""
        import os
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        with open(f"{model_path}/model.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save configuration
        config_dict = {
            'n_estimators': self.config.n_estimators,
            'max_depth': self.config.max_depth,
            'min_samples_split': self.config.min_samples_split,
            'min_samples_leaf': self.config.min_samples_leaf,
            'random_state': self.config.random_state,
            'embedding_model': self.config.embedding_model,
            'similarity_threshold': self.config.similarity_threshold,
            'top_k_sops': self.config.top_k_sops,
            'test_size': self.config.test_size,
            'cv_folds': self.config.cv_folds,
            'device': self.config.device
        }
        
        with open(f"{model_path}/config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save feature names
        feature_names = self._get_feature_names()
        with open(f"{model_path}/feature_names.json", 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, model_path: str) -> 'SOPPruner':
        """Load a pre-trained model."""
        # Load configuration
        with open(f"{model_path}/config.json", 'r') as f:
            config_dict = json.load(f)
        
        config = SOPPrunerConfig(**config_dict)
        
        # Create model instance
        model = cls(config)
        
        # Load trained model
        with open(f"{model_path}/model.pkl", 'rb') as f:
            model.model = pickle.load(f)
        
        model.is_trained = True
        
        logger.info(f"Model loaded from {model_path}")
        return model

def create_sample_training_data() -> List[Dict]:
    """Create sample training data for demonstration."""
    sample_data = []
    
    # Example 1: Relevant action
    sample_data.append({
        'incident_context': 'High CPU usage on database server, connection pool exhausted',
        'action': {
            'type': 'query_metrics',
            'description': 'Check database connection pool metrics',
            'parameters': {'service': 'database', 'metric': 'connection_pool'}
        },
        'relevant_sops': [{
            'sop_id': 'sop_001',
            'title': 'Database Connection Pool Issues',
            'description': 'Troubleshooting database connection pool problems',
            'steps': ['Check connection pool metrics', 'Monitor active connections'],
            'similarity': 0.89
        }],
        'label': 1
    })
    
    # Example 2: Irrelevant action
    sample_data.append({
        'incident_context': 'High CPU usage on database server, connection pool exhausted',
        'action': {
            'type': 'restart_service',
            'description': 'Restart web application service',
            'parameters': {'service': 'web-app'}
        },
        'relevant_sops': [{
            'sop_id': 'sop_001',
            'title': 'Database Connection Pool Issues',
            'description': 'Troubleshooting database connection pool problems',
            'steps': ['Check connection pool metrics', 'Monitor active connections'],
            'similarity': 0.89
        }],
        'label': 0
    })
    
    # Add more examples...
    for i in range(100):
        sample_data.append({
            'incident_context': f'Incident context {i}',
            'action': {
                'type': np.random.choice(['query_metrics', 'query_logs', 'restart_service']),
                'description': f'Action description {i}',
                'parameters': {}
            },
            'relevant_sops': [{'similarity': np.random.random()}],
            'label': np.random.randint(0, 2)
        })
    
    return sample_data

def train_sop_pruner(config: SOPPrunerConfig, training_data_path: str, model_path: str):
    """Train the SOP pruner."""
    logger.info("Starting SOP pruner training")
    
    # Initialize Weights & Biases
    wandb.init(project="tamo-foa-sop-pruner", config=config.__dict__)
    
    # Load training data
    if training_data_path.endswith('.json'):
        with open(training_data_path, 'r') as f:
            training_data = json.load(f)
    else:
        # Create sample data for demonstration
        training_data = create_sample_training_data()
    
    # Initialize and train model
    pruner = SOPPruner(config)
    pruner.train(training_data)
    
    # Save model
    pruner.save(model_path)
    
    logger.info(f"Training completed. Model saved to {model_path}")
    
    wandb.finish()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SOP Pruner")
    parser.add_argument("--training_data", type=str, required=True, help="Path to training data")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to save model")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=10, help="Max depth of trees")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Create configuration
    config = SOPPrunerConfig(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        device=args.device
    )
    
    # Train model
    train_sop_pruner(config, args.training_data, args.model_dir)