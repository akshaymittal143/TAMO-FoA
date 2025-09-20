"""
HDM-2 Hallucination Detector for TAMO-FoA Framework

This module implements the HDM-2 (Hallucination Detection Module-2) that provides
enterprise-grade verification for LLM outputs in AIOps contexts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import wandb
from sentence_transformers import SentenceTransformer
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HDM2Config:
    """Configuration for the HDM-2 detector."""
    # Model parameters
    model_name: str = "microsoft/deberta-base"
    max_length: int = 512
    num_labels: int = 2  # 0: factual, 1: hallucinated
    
    # Training parameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 10
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Contrastive learning parameters
    contrastive_temperature: float = 0.07
    contrastive_weight: float = 0.3
    
    # Verification parameters
    confidence_threshold: float = 0.7
    verification_methods: List[str] = None  # Will be set to ["context", "knowledge", "consistency"]
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.verification_methods is None:
            self.verification_methods = ["context", "knowledge", "consistency"]

class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning discriminative representations."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            features: Feature representations (batch_size, feature_dim)
            labels: Binary labels (batch_size,)
        """
        batch_size = features.size(0)
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same label)
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=features.device)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # Average over positive pairs
        loss = -(mask * log_prob).sum() / mask.sum()
        
        return loss

class HDM2Detector(nn.Module):
    """HDM-2 Hallucination Detection Module."""
    
    def __init__(self, config: HDM2Config):
        super().__init__()
        self.config = config
        
        # Main verification model
        self.verification_model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, num_labels=config.num_labels
        )
        
        # Contrastive learning components
        self.feature_extractor = AutoModel.from_pretrained(config.model_name)
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.feature_extractor.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Context verification head
        self.context_verifier = nn.Linear(
            self.feature_extractor.config.hidden_size, 1
        )
        
        # Knowledge verification head
        self.knowledge_verifier = nn.Linear(
            self.feature_extractor.config.hidden_size, 1
        )
        
        # Consistency verification head
        self.consistency_verifier = nn.Linear(
            self.feature_extractor.config.hidden_size, 1
        )
        
        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss(config.contrastive_temperature)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for HDM-2 detector.
        
        Args:
            input_ids: Tokenized input (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Ground truth labels (batch_size,)
            
        Returns:
            Dictionary containing predictions and losses
        """
        # Get features from pre-trained model
        features = self.feature_extractor(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        
        # Global average pooling
        pooled_features = features.mean(dim=1)
        
        # Main verification prediction
        verification_logits = self.verification_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).logits
        
        # Individual verification heads
        context_scores = torch.sigmoid(self.context_verifier(pooled_features))
        knowledge_scores = torch.sigmoid(self.knowledge_verifier(pooled_features))
        consistency_scores = torch.sigmoid(self.consistency_verifier(pooled_features))
        
        # Contrastive features
        contrastive_features = self.contrastive_head(pooled_features)
        
        outputs = {
            'verification_logits': verification_logits,
            'context_scores': context_scores,
            'knowledge_scores': knowledge_scores,
            'consistency_scores': consistency_scores,
            'contrastive_features': contrastive_features
        }
        
        if labels is not None:
            # Main verification loss
            verification_loss = F.cross_entropy(verification_logits, labels)
            
            # Contrastive loss
            contrastive_loss = self.contrastive_loss(contrastive_features, labels)
            
            # Individual verification losses
            context_loss = F.binary_cross_entropy(context_scores.squeeze(), labels.float())
            knowledge_loss = F.binary_cross_entropy(knowledge_scores.squeeze(), labels.float())
            consistency_loss = F.binary_cross_entropy(consistency_scores.squeeze(), labels.float())
            
            # Combined loss
            total_loss = (verification_loss + 
                         self.config.contrastive_weight * contrastive_loss +
                         0.1 * (context_loss + knowledge_loss + consistency_loss))
            
            outputs['loss'] = total_loss
            outputs['verification_loss'] = verification_loss
            outputs['contrastive_loss'] = contrastive_loss
            outputs['context_loss'] = context_loss
            outputs['knowledge_loss'] = knowledge_loss
            outputs['consistency_loss'] = consistency_loss
        
        return outputs
    
    def verify(self, response: str, context: str, method: str = "all") -> Dict[str, Any]:
        """
        Verify a response against context.
        
        Args:
            response: LLM response to verify
            context: Context information
            method: Verification method ("all", "context", "knowledge", "consistency")
            
        Returns:
            Verification results
        """
        self.eval()
        
        # Prepare input
        if method == "context":
            input_text = f"Context: {context}\nResponse: {response}\nIs the response consistent with the context?"
        elif method == "knowledge":
            input_text = f"Response: {response}\nIs this response factually correct?"
        elif method == "consistency":
            input_text = f"Response: {response}\nIs this response internally consistent?"
        else:  # "all"
            input_text = f"Context: {context}\nResponse: {response}\nIs the response accurate and consistent?"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self(**inputs)
            
            # Get predictions
            verification_probs = F.softmax(outputs['verification_logits'], dim=-1)
            verification_pred = torch.argmax(verification_probs, dim=-1)
            
            # Get individual scores
            context_score = outputs['context_scores'].item()
            knowledge_score = outputs['knowledge_scores'].item()
            consistency_score = outputs['consistency_scores'].item()
            
            # Overall confidence
            confidence = verification_probs[0][0].item()  # Probability of being factual
            
            result = {
                'is_factual': verification_pred.item() == 0,
                'confidence': confidence,
                'context_score': context_score,
                'knowledge_score': knowledge_score,
                'consistency_score': consistency_score,
                'verification_method': method
            }
            
            return result
    
    def batch_verify(self, responses: List[str], contexts: List[str]) -> List[Dict[str, Any]]:
        """Verify multiple responses in batch."""
        self.eval()
        
        results = []
        for response, context in zip(responses, contexts):
            result = self.verify(response, context)
            results.append(result)
        
        return results

class HallucinationDataset(Dataset):
    """Dataset for hallucination detection training."""
    
    def __init__(self, data_path: str, split: str = "train"):
        self.data_path = data_path
        self.split = split
        
        # Load data
        with open(f"{data_path}/{split}.json", 'r') as f:
            self.data = json.load(f)
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Prepare input text
        if 'context' in sample:
            input_text = f"Context: {sample['context']}\nResponse: {sample['response']}\nIs the response accurate?"
        else:
            input_text = f"Response: {sample['response']}\nIs this response factually correct?"
        
        return {
            'text': input_text,
            'label': sample['label'],  # 0: factual, 1: hallucinated
            'response': sample['response'],
            'context': sample.get('context', '')
        }

class HDM2Trainer:
    """Custom trainer for HDM-2 detector."""
    
    def __init__(self, model: HDM2Detector, config: HDM2Config):
        self.model = model
        self.config = config
        self.tokenizer = model.tokenizer
        
    def train(self, train_dataset: Dataset, val_dataset: Dataset, model_path: str):
        """Train the HDM-2 detector."""
        logger.info("Starting HDM-2 detector training")
        
        # Initialize Weights & Biases
        wandb.init(project="tamo-foa-hdm2", config=self.config.__dict__)
        
        # Prepare data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        # Initialize optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * self.config.epochs
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                # Tokenize
                inputs = self.tokenizer(
                    batch['text'],
                    return_tensors="pt",
                    max_length=self.config.max_length,
                    truncation=True,
                    padding=True
                ).to(self.config.device)
                
                labels = torch.tensor(batch['label'], dtype=torch.long).to(self.config.device)
                
                # Forward pass
                outputs = self.model(**inputs, labels=labels)
                loss = outputs['loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            # Validation phase
            val_loss, val_metrics = self._evaluate(val_loader)
            
            # Log metrics
            avg_train_loss = train_loss / len(train_loader)
            
            logger.info(f"Epoch {epoch+1}/{self.config.epochs}: "
                       f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics['accuracy'],
                "val_f1": val_metrics['f1'],
                "learning_rate": scheduler.get_last_lr()[0]
            })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save(model_path)
                logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        logger.info(f"Training completed. Best model saved to {model_path}")
        
        wandb.finish()
    
    def _evaluate(self, data_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Tokenize
                inputs = self.tokenizer(
                    batch['text'],
                    return_tensors="pt",
                    max_length=self.config.max_length,
                    truncation=True,
                    padding=True
                ).to(self.config.device)
                
                labels = torch.tensor(batch['label'], dtype=torch.long).to(self.config.device)
                
                # Forward pass
                outputs = self.model(**inputs, labels=labels)
                loss = outputs['loss']
                
                # Get predictions
                predictions = torch.argmax(outputs['verification_logits'], dim=-1)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        avg_loss = total_loss / len(data_loader)
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return avg_loss, metrics

def create_sample_training_data() -> Dict[str, List[Dict]]:
    """Create sample training data for demonstration."""
    sample_data = {
        'train': [],
        'val': []
    }
    
    # Factual examples
    factual_examples = [
        {
            'context': 'Database connection pool is at 95% capacity',
            'response': 'The database connection pool is nearly full at 95% capacity',
            'label': 0
        },
        {
            'context': 'CPU usage on web server is 80%',
            'response': 'High CPU utilization detected on the web server',
            'label': 0
        },
        {
            'context': 'Service is running on port 8080',
            'response': 'The application is listening on port 8080',
            'label': 0
        }
    ]
    
    # Hallucinated examples
    hallucinated_examples = [
        {
            'context': 'Database connection pool is at 95% capacity',
            'response': 'The database connection pool is completely empty and needs more connections',
            'label': 1
        },
        {
            'context': 'CPU usage on web server is 80%',
            'response': 'The web server is completely idle with 0% CPU usage',
            'label': 1
        },
        {
            'context': 'Service is running on port 8080',
            'response': 'The application is not running and port 8080 is closed',
            'label': 1
        }
    ]
    
    # Add more examples
    for i in range(50):
        if i % 2 == 0:
            sample_data['train'].extend(factual_examples)
            sample_data['val'].extend(hallucinated_examples)
        else:
            sample_data['train'].extend(hallucinated_examples)
            sample_data['val'].extend(factual_examples)
    
    return sample_data

def train_hdm2_detector(config: HDM2Config, data_path: str, model_path: str):
    """Train the HDM-2 detector."""
    logger.info("Starting HDM-2 detector training")
    
    # Create sample data if no data path provided
    if data_path is None:
        sample_data = create_sample_training_data()
        
        # Save sample data
        import os
        os.makedirs("sample_data", exist_ok=True)
        with open("sample_data/train.json", 'w') as f:
            json.dump(sample_data['train'], f, indent=2)
        with open("sample_data/val.json", 'w') as f:
            json.dump(sample_data['val'], f, indent=2)
        
        data_path = "sample_data"
    
    # Load datasets
    train_dataset = HallucinationDataset(data_path, "train")
    val_dataset = HallucinationDataset(data_path, "val")
    
    # Initialize model
    model = HDM2Detector(config)
    
    # Initialize trainer
    trainer = HDM2Trainer(model, config)
    
    # Train model
    trainer.train(train_dataset, val_dataset, model_path)
    
    logger.info(f"Training completed. Model saved to {model_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train HDM-2 Detector")
    parser.add_argument("--data_dir", type=str, help="Path to training data")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to save model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Create configuration
    config = HDM2Config(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        device=args.device
    )
    
    # Train model
    train_hdm2_detector(config, args.data_dir, args.model_dir)
