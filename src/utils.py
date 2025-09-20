"""
Utility functions for TAMO-FoA Framework

This module provides common utilities for data processing, evaluation,
and system configuration across the TAMO-FoA components.
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create a dummy torch module for basic functionality
    class DummyTorch:
        def __init__(self):
            self.device = self
            self.cuda = self
            
        def is_available(self):
            return False
            
        def get_device_name(self):
            return "CPU (PyTorch not available)"
            
        def manual_seed(self, seed):
            pass
            
        def cuda_manual_seed_all(self, seed):
            pass
            
        def backends(self):
            class Cudnn:
                deterministic = True
                benchmark = False
            return Cudnn()
    
    torch = DummyTorch()

import numpy as np
import pandas as pd
import json
import logging
import os
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import yaml
from pathlib import Path
import hashlib
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemConfig:
    """System configuration for TAMO-FoA."""
    # Model paths
    encoder_model_path: str = "models/encoder"
    sop_pruner_path: str = "models/sop_pruner"
    hdm2_model_path: str = "models/hdm2"
    
    # Data paths
    data_dir: str = "data"
    cache_dir: str = "cache"
    
    # Neo4j configuration
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    
    # Kafka configuration
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "tamo-foa-incidents"
    
    # Performance settings
    max_concurrent_incidents: int = 10
    response_timeout: int = 30
    memory_limit_gb: int = 8
    
    # Device settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    @classmethod
    def load(cls, config_path: str) -> 'SystemConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def save(self, config_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'encoder_model_path': self.encoder_model_path,
            'sop_pruner_path': self.sop_pruner_path,
            'hdm2_model_path': self.hdm2_model_path,
            'data_dir': self.data_dir,
            'cache_dir': self.cache_dir,
            'neo4j_uri': self.neo4j_uri,
            'neo4j_user': self.neo4j_user,
            'neo4j_password': self.neo4j_password,
            'redis_host': self.redis_host,
            'redis_port': self.redis_port,
            'redis_db': self.redis_db,
            'kafka_bootstrap_servers': self.kafka_bootstrap_servers,
            'kafka_topic': self.kafka_topic,
            'max_concurrent_incidents': self.max_concurrent_incidents,
            'response_timeout': self.response_timeout,
            'memory_limit_gb': self.memory_limit_gb,
            'device': self.device,
            'num_workers': self.num_workers
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda_manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """Get the appropriate device for computation."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        if TORCH_AVAILABLE:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        else:
            device = "cpu"
            logger.info("Using CPU device (PyTorch not available)")
    
    return device

def load_json(file_path: str) -> Dict:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: Dict, file_path: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_pickle(file_path: str) -> Any:
    """Load pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data: Any, file_path: str):
    """Save data to pickle file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics

def format_confidence_interval(mean: float, std: float, n: int = 1) -> str:
    """Format confidence interval for reporting."""
    import scipy.stats as stats
    
    # Calculate 95% confidence interval
    ci = stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
    return f"{mean:.1f}±{std:.1f}"

def create_directory_structure(base_path: str):
    """Create the standard TAMO-FoA directory structure."""
    directories = [
        "data/OpenRCA",
        "data/ITBench", 
        "data/AIOpsLab",
        "data/CloudDiagBench",
        "models/encoder",
        "models/sop_pruner",
        "models/hdm2",
        "notebooks",
        "experiments",
        "logs",
        "cache"
    ]
    
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"Created directory: {full_path}")

def validate_data_format(data: Dict) -> bool:
    """Validate that data follows the expected format."""
    required_fields = ['metrics', 'logs', 'traces']
    
    for field in required_fields:
        if field not in data:
            logger.error(f"Missing required field: {field}")
            return False
    
    # Validate metrics format
    if not isinstance(data['metrics'], (list, np.ndarray, torch.Tensor)):
        logger.error("Metrics must be a list, numpy array, or torch tensor")
        return False
    
    # Validate logs format
    if not isinstance(data['logs'], list):
        logger.error("Logs must be a list of strings")
        return False
    
    # Validate traces format
    if not isinstance(data['traces'], (list, dict)):
        logger.error("Traces must be a list or dictionary")
        return False
    
    return True

def preprocess_metrics(metrics: Union[List, np.ndarray], 
                      normalize: bool = True):
    """Preprocess metrics data."""
    if isinstance(metrics, list):
        metrics = np.array(metrics)
    
    if TORCH_AVAILABLE and isinstance(metrics, np.ndarray):
        metrics = torch.from_numpy(metrics).float()
    elif isinstance(metrics, np.ndarray):
        metrics = metrics.astype(np.float32)
    
    # Normalize if requested
    if normalize:
        mean_val = np.mean(metrics) if isinstance(metrics, np.ndarray) else metrics.mean()
        std_val = np.std(metrics) if isinstance(metrics, np.ndarray) else metrics.std()
        metrics = (metrics - mean_val) / (std_val + 1e-8)
    
    return metrics

def preprocess_logs(logs: List[str], max_length: int = 512) -> List[str]:
    """Preprocess log data."""
    processed_logs = []
    
    for log in logs:
        # Truncate if too long
        if len(log) > max_length:
            log = log[:max_length]
        
        # Basic cleaning
        log = log.strip()
        processed_logs.append(log)
    
    return processed_logs

def preprocess_traces(traces: List[Dict]) -> List[Dict]:
    """Preprocess trace data."""
    processed_traces = []
    
    for trace in traces:
        # Ensure required fields exist
        if 'nodes' not in trace:
            trace['nodes'] = []
        if 'edges' not in trace:
            trace['edges'] = []
        
        # Validate node format
        if trace['nodes'] and not isinstance(trace['nodes'][0], (list, tuple)):
            logger.warning("Trace nodes should be lists/tuples of features")
        
        # Validate edge format
        if trace['edges'] and len(trace['edges'][0]) != 2:
            logger.warning("Trace edges should be pairs of node indices")
        
        processed_traces.append(trace)
    
    return processed_traces

class PerformanceMonitor:
    """Monitor system performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, name: str):
        """Start timing an operation."""
        import time
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration."""
        import time
        if name not in self.start_times:
            logger.warning(f"Timer {name} was not started")
            return 0.0
        
        duration = time.time() - self.start_times[name]
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(duration)
        del self.start_times[name]
        return duration
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric."""
        if name not in self.metrics:
            return {}
        
        values = self.metrics[name]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    
    def log_performance(self):
        """Log all performance metrics."""
        logger.info("Performance Metrics:")
        for name in self.metrics:
            stats = self.get_stats(name)
            logger.info(f"  {name}: {stats['mean']:.3f}±{stats['std']:.3f}s "
                       f"(min: {stats['min']:.3f}s, max: {stats['max']:.3f}s, "
                       f"count: {stats['count']})")

class DataValidator:
    """Validate data integrity and format."""
    
    @staticmethod
    def validate_incident_data(incident: Dict) -> Tuple[bool, List[str]]:
        """Validate incident data format."""
        errors = []
        
        # Check required fields
        required_fields = ['incident_id', 'timestamp', 'severity', 'description']
        for field in required_fields:
            if field not in incident:
                errors.append(f"Missing required field: {field}")
        
        # Validate timestamp
        if 'timestamp' in incident:
            try:
                pd.to_datetime(incident['timestamp'])
            except:
                errors.append("Invalid timestamp format")
        
        # Validate severity
        if 'severity' in incident:
            valid_severities = ['low', 'medium', 'high', 'critical']
            if incident['severity'].lower() not in valid_severities:
                errors.append(f"Invalid severity: {incident['severity']}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_telemetry_data(telemetry: Dict) -> Tuple[bool, List[str]]:
        """Validate telemetry data format."""
        errors = []
        
        # Check required fields
        required_fields = ['metrics', 'logs', 'traces']
        for field in required_fields:
            if field not in telemetry:
                errors.append(f"Missing required field: {field}")
        
        # Validate metrics
        if 'metrics' in telemetry:
            metrics = telemetry['metrics']
            if not isinstance(metrics, (list, dict)):
                errors.append("Metrics must be a list or dictionary")
        
        # Validate logs
        if 'logs' in telemetry:
            logs = telemetry['logs']
            if not isinstance(logs, list):
                errors.append("Logs must be a list")
            elif logs and not all(isinstance(log, str) for log in logs):
                errors.append("All logs must be strings")
        
        # Validate traces
        if 'traces' in telemetry:
            traces = telemetry['traces']
            if not isinstance(traces, (list, dict)):
                errors.append("Traces must be a list or dictionary")
        
        return len(errors) == 0, errors

def generate_incident_hash(incident: Dict) -> str:
    """Generate a unique hash for an incident."""
    # Create a string representation of the incident
    incident_str = json.dumps(incident, sort_keys=True)
    
    # Generate SHA-256 hash
    return hashlib.sha256(incident_str.encode()).hexdigest()[:16]

def load_sample_data() -> Dict[str, Any]:
    """Load sample data for testing and demonstration."""
    sample_data = {
        'incidents': [
            {
                'incident_id': 'inc_001',
                'timestamp': '2024-01-15T10:30:00Z',
                'severity': 'high',
                'description': 'Database connection pool exhaustion',
                'service': 'OrderService',
                'metrics': {
                    'cpu_usage': 85.2,
                    'memory_usage': 78.5,
                    'connection_pool': 95.0
                },
                'logs': [
                    'ERROR: Connection pool exhausted',
                    'WARN: High connection count detected',
                    'INFO: Service restart initiated'
                ],
                'traces': {
                    'nodes': [
                        [0.1, 0.2, 0.3],  # Service node features
                        [0.4, 0.5, 0.6]   # Database node features
                    ],
                    'edges': [
                        [0, 1]  # Connection between service and database
                    ]
                }
            }
        ],
        'sops': [
            {
                'sop_id': 'sop_001',
                'title': 'Database Connection Pool Issues',
                'description': 'Troubleshooting database connection pool problems',
                'steps': [
                    'Check connection pool metrics',
                    'Monitor active connections',
                    'Review application configuration',
                    'Restart service if necessary'
                ],
                'category': 'database',
                'tags': ['database', 'connections', 'pool']
            }
        ]
    }
    
    return sample_data

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

if __name__ == "__main__":
    # Example usage
    setup_logging()
    
    # Create sample data
    sample_data = load_sample_data()
    
    # Validate data
    validator = DataValidator()
    for incident in sample_data['incidents']:
        is_valid, errors = validator.validate_incident_data(incident)
        if is_valid:
            logger.info(f"Incident {incident['incident_id']} is valid")
        else:
            logger.error(f"Incident {incident['incident_id']} has errors: {errors}")
    
    # Performance monitoring example
    monitor = PerformanceMonitor()
    monitor.start_timer("sample_operation")
    
    # Simulate some work
    import time
    time.sleep(0.1)
    
    duration = monitor.end_timer("sample_operation")
    logger.info(f"Operation took {duration:.3f} seconds")
    
    # Log performance
    monitor.log_performance()
