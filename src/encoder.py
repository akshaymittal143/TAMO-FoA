"""
Multi-Modal Encoder for TAMO-FoA Framework

This module implements the hybrid multi-modal encoder that processes three data modalities:
1. Metrics (time-series) using 1D-CNN
2. Logs (textual) using BERT+LSTM
3. Traces (graph-structured) using GNN

The encoded representations are fused using a diffusion-based process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import wandb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EncoderConfig:
    """Configuration for the multi-modal encoder."""
    # Model dimensions
    metrics_dim: int = 100
    logs_dim: int = 768
    traces_dim: int = 256
    hidden_dim: int = 1024
    output_dim: int = 1024
    
    # Architecture parameters
    cnn_layers: int = 3
    cnn_kernel_size: int = 3
    lstm_layers: int = 2
    lstm_hidden: int = 256
    gnn_layers: int = 2
    gnn_hidden: int = 128
    
    # Diffusion parameters
    diffusion_steps: int = 1000
    diffusion_noise_schedule: str = "cosine"
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    weight_decay: float = 1e-5
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class MetricsEncoder(nn.Module):
    """1D CNN encoder for time-series metrics."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 1D CNN layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim),
        ])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input metrics tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Encoded metrics tensor of shape (batch_size, output_dim)
        """
        # Transpose for Conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = F.relu(bn(conv(x)))
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Global average pooling
        x = self.pool(x).squeeze(-1)  # (batch_size, hidden_dim)
        
        # Final linear layer
        x = self.fc(x)
        return x

class LogsEncoder(nn.Module):
    """BERT + LSTM encoder for textual logs."""
    
    def __init__(self, bert_model_name: str = "bert-base-uncased", 
                 lstm_hidden: int = 256, output_dim: int = 768):
        super().__init__()
        self.bert_model_name = bert_model_name
        self.lstm_hidden = lstm_hidden
        self.output_dim = output_dim
        
        # Load pre-trained BERT
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # LSTM for sequence processing
        self.lstm = nn.LSTM(
            input_size=self.bert.config.hidden_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Output projection
        self.fc = nn.Linear(lstm_hidden * 2, output_dim)  # *2 for bidirectional
        
    def forward(self, log_texts: List[str]) -> torch.Tensor:
        """
        Args:
            log_texts: List of log text strings
        Returns:
            Encoded logs tensor of shape (batch_size, output_dim)
        """
        # Tokenize and encode with BERT
        inputs = self.tokenizer(
            log_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(next(self.parameters()).device)
        
        # Get BERT embeddings
        with torch.no_grad():
            bert_outputs = self.bert(**inputs)
            bert_embeddings = bert_outputs.last_hidden_state
        
        # Process with LSTM
        lstm_output, (hidden, cell) = self.lstm(bert_embeddings)
        
        # Use the last hidden state (both directions)
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # Final projection
        output = self.fc(last_hidden)
        return output

class TracesEncoder(nn.Module):
    """Graph Neural Network encoder for distributed traces."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # GCN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output projection
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features of shape (num_nodes, input_dim)
            edge_index: Edge connectivity of shape (2, num_edges)
            batch: Batch assignment for each node
        Returns:
            Encoded traces tensor of shape (batch_size, output_dim)
        """
        # Apply GCN layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Final projection
        x = self.fc(x)
        return x

class DiffusionFusion(nn.Module):
    """Diffusion-based fusion module for multi-modal representations."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_timesteps: int = 1000):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_timesteps = num_timesteps
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Denoising network
        self.denoise_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal timestep embeddings."""
        half_dim = 128 // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_embed(emb)
    
    def forward(self, metrics_enc: torch.Tensor, logs_enc: torch.Tensor, 
                traces_enc: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            metrics_enc: Encoded metrics (batch_size, hidden_dim)
            logs_enc: Encoded logs (batch_size, hidden_dim)
            traces_enc: Encoded traces (batch_size, hidden_dim)
            timesteps: Diffusion timesteps (batch_size,)
        Returns:
            Fused representation (batch_size, output_dim)
        """
        # Concatenate multi-modal representations
        x = torch.cat([metrics_enc, logs_enc, traces_enc], dim=-1)
        
        # Get timestep embedding
        t_emb = self.get_timestep_embedding(timesteps)
        
        # Combine input with timestep embedding
        x = torch.cat([x, t_emb], dim=-1)
        
        # Denoising network
        output = self.denoise_net(x)
        return output

class MultiModalEncoder(nn.Module):
    """Main multi-modal encoder combining all modalities."""
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        
        # Individual encoders
        self.metrics_encoder = MetricsEncoder(
            input_dim=config.metrics_dim,
            hidden_dim=config.hidden_dim // 3,
            output_dim=config.hidden_dim // 3
        )
        
        self.logs_encoder = LogsEncoder(
            lstm_hidden=config.lstm_hidden,
            output_dim=config.hidden_dim // 3
        )
        
        self.traces_encoder = TracesEncoder(
            input_dim=config.traces_dim,
            hidden_dim=config.gnn_hidden,
            output_dim=config.hidden_dim // 3,
            num_layers=config.gnn_layers
        )
        
        # Diffusion fusion
        self.fusion = DiffusionFusion(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            num_timesteps=config.diffusion_steps
        )
        
        # Noise schedule for diffusion
        self.register_buffer('betas', self._get_noise_schedule())
        
    def _get_noise_schedule(self) -> torch.Tensor:
        """Generate cosine noise schedule for diffusion."""
        def cosine_beta_schedule(timesteps, s=0.008):
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        
        return cosine_beta_schedule(self.config.diffusion_steps)
    
    def forward(self, metrics: torch.Tensor, logs: List[str], 
                traces_data: List[Data], timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            metrics: Metrics tensor (batch_size, seq_len, metrics_dim)
            logs: List of log text strings
            traces_data: List of PyG Data objects for traces
            timesteps: Optional diffusion timesteps
        Returns:
            Fused multi-modal representation (batch_size, output_dim)
        """
        # Encode each modality
        metrics_enc = self.metrics_encoder(metrics)
        logs_enc = self.logs_encoder(logs)
        
        # Process traces (batch them if needed)
        if traces_data:
            traces_batch = Batch.from_data_list(traces_data)
            traces_enc = self.traces_encoder(
                traces_batch.x, traces_batch.edge_index, traces_batch.batch
            )
        else:
            # Create zero tensor if no traces
            traces_enc = torch.zeros(metrics.size(0), self.config.hidden_dim // 3, 
                                   device=metrics.device)
        
        # Generate random timesteps if not provided
        if timesteps is None:
            timesteps = torch.randint(0, self.config.diffusion_steps, 
                                    (metrics.size(0),), device=metrics.device)
        
        # Diffusion-based fusion
        fused_representation = self.fusion(
            torch.cat([metrics_enc, logs_enc, traces_enc], dim=-1),
            timesteps
        )
        
        return fused_representation
    
    def compute_diffusion_loss(self, x: torch.Tensor, noise: torch.Tensor, 
                              timesteps: torch.Tensor) -> torch.Tensor:
        """Compute diffusion loss for training."""
        # Add noise to input
        noisy_x = x + noise
        
        # Predict noise
        predicted_noise = self.fusion(noisy_x, timesteps)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        return loss
    
    @classmethod
    def load(cls, model_path: str, config_path: Optional[str] = None) -> 'MultiModalEncoder':
        """Load pre-trained model."""
        if config_path is None:
            config_path = f"{model_path}/config.json"
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = EncoderConfig(**config_dict)
        model = cls(config)
        model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location='cpu'))
        model.eval()
        
        return model
    
    def save(self, model_path: str):
        """Save model and configuration."""
        import os
        os.makedirs(model_path, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), f"{model_path}/model.pt")
        
        # Save configuration
        config_dict = {
            'metrics_dim': self.config.metrics_dim,
            'logs_dim': self.config.logs_dim,
            'traces_dim': self.config.traces_dim,
            'hidden_dim': self.config.hidden_dim,
            'output_dim': self.config.output_dim,
            'cnn_layers': self.config.cnn_layers,
            'cnn_kernel_size': self.config.cnn_kernel_size,
            'lstm_layers': self.config.lstm_layers,
            'lstm_hidden': self.config.lstm_hidden,
            'gnn_layers': self.config.gnn_layers,
            'gnn_hidden': self.config.gnn_hidden,
            'diffusion_steps': self.config.diffusion_steps,
            'learning_rate': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'epochs': self.config.epochs,
            'weight_decay': self.config.weight_decay,
            'device': self.config.device
        }
        
        with open(f"{model_path}/config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

class MultiModalDataset(Dataset):
    """Dataset for multi-modal training data."""
    
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
        
        # Extract features
        metrics = torch.tensor(sample['metrics'], dtype=torch.float32)
        logs = sample['logs']
        traces = self._parse_traces(sample['traces'])
        
        return {
            'metrics': metrics,
            'logs': logs,
            'traces': traces,
            'label': sample.get('label', 0)
        }
    
    def _parse_traces(self, traces_data: Dict) -> Data:
        """Parse trace data into PyG Data object."""
        nodes = torch.tensor(traces_data['nodes'], dtype=torch.float32)
        edges = torch.tensor(traces_data['edges'], dtype=torch.long).T
        
        return Data(x=nodes, edge_index=edges)

def train_encoder(config: EncoderConfig, data_path: str, model_path: str):
    """Train the multi-modal encoder."""
    logger.info("Starting multi-modal encoder training")
    
    # Initialize model
    model = MultiModalEncoder(config).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, 
                          weight_decay=config.weight_decay)
    
    # Initialize Weights & Biases
    wandb.init(project="tamo-foa-encoder", config=config.__dict__)
    
    # Load datasets
    train_dataset = MultiModalDataset(data_path, "train")
    val_dataset = MultiModalDataset(data_path, "val")
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Move data to device
            metrics = batch['metrics'].to(config.device)
            logs = batch['logs']
            traces = [t.to(config.device) for t in batch['traces']]
            
            # Generate random timesteps and noise
            timesteps = torch.randint(0, config.diffusion_steps, 
                                    (metrics.size(0),), device=config.device)
            noise = torch.randn_like(metrics)
            
            # Forward pass
            fused_repr = model(metrics, logs, traces, timesteps)
            loss = model.compute_diffusion_loss(metrics, noise, timesteps)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                metrics = batch['metrics'].to(config.device)
                logs = batch['logs']
                traces = [t.to(config.device) for t in batch['traces']]
                
                timesteps = torch.randint(0, config.diffusion_steps, 
                                        (metrics.size(0),), device=config.device)
                noise = torch.randn_like(metrics)
                
                fused_repr = model(metrics, logs, traces, timesteps)
                loss = model.compute_diffusion_loss(metrics, noise, timesteps)
                
                val_loss += loss.item()
        
        # Log metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{config.epochs}: "
                   f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            model.save(f"{model_path}/checkpoint_{epoch+1}")
    
    # Save final model
    model.save(model_path)
    logger.info(f"Training completed. Model saved to {model_path}")
    
    wandb.finish()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Multi-Modal Encoder")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to training data")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to save model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Create configuration
    config = EncoderConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        device=args.device
    )
    
    # Train model
    train_encoder(config, args.data_dir, args.model_dir)
