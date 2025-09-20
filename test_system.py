#!/usr/bin/env python3
"""
TAMO-FoA System Test Script

This script performs basic tests to ensure all components work correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from encoder import MultiModalDiffusionEncoder, EncoderConfig
        print("✅ Encoder module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import encoder: {e}")
        return False
    
    try:
        from sop_pruner import SOPPruner, SOPPrunerConfig
        print("✅ SOP pruner module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import SOP pruner: {e}")
        return False
    
    try:
        from hdm2_detector import HDM2Detector, HDM2Config
        print("✅ HDM-2 detector module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import HDM-2 detector: {e}")
        return False
    
    try:
        from utils import SystemConfig, set_seed, get_device
        print("✅ Utils module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import utils: {e}")
        return False
    
    try:
        from main import TAMOFoA
        print("✅ Main module imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import main: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from utils import SystemConfig
        config = SystemConfig()
        print("✅ SystemConfig created successfully")
        print(f"   Device: {config.device}")
        print(f"   Data dir: {config.data_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to create SystemConfig: {e}")
        return False

def test_encoder():
    """Test encoder initialization."""
    print("\nTesting encoder...")
    
    try:
        from encoder import MultiModalDiffusionEncoder, EncoderConfig
        
        config = EncoderConfig(
            num_metrics=5,
            metric_seq_len=50,
            log_model_name='bert-base-uncased',
            node_feature_dim=64,
            hidden_dim=256,
            diffusion_steps=100
        )
        
        encoder = MultiModalDiffusionEncoder(
            num_metrics=config.num_metrics,
            metric_seq_len=config.metric_seq_len,
            log_model_name=config.log_model_name,
            node_feature_dim=config.node_feature_dim,
            hidden_dim=config.hidden_dim,
            diffusion_steps=config.diffusion_steps
        )
        
        print("✅ Encoder initialized successfully")
        print(f"   Parameters: {sum(p.numel() for p in encoder.parameters())}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize encoder: {e}")
        return False

def test_sop_pruner():
    """Test SOP pruner initialization."""
    print("\nTesting SOP pruner...")
    
    try:
        from sop_pruner import SOPPruner, SOPPrunerConfig
        
        config = SOPPrunerConfig(
            max_sops=100,
            similarity_threshold=0.7,
            embedding_model='sentence-transformers/all-MiniLM-L6-v2',
            max_actions=20
        )
        
        pruner = SOPPruner(config)
        print("✅ SOP pruner initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize SOP pruner: {e}")
        return False

def test_hdm2_detector():
    """Test HDM-2 detector initialization."""
    print("\nTesting HDM-2 detector...")
    
    try:
        from hdm2_detector import HDM2Detector, HDM2Config
        
        config = HDM2Config(
            model_name='microsoft/deberta-base',
            max_length=256,
            learning_rate=2e-5,
            batch_size=8,
            num_epochs=5,
            dropout=0.1
        )
        
        detector = HDM2Detector(config)
        print("✅ HDM-2 detector initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize HDM-2 detector: {e}")
        return False

def test_main_system():
    """Test main TAMO-FoA system."""
    print("\nTesting main TAMO-FoA system...")
    
    try:
        from main import TAMOFoA
        
        # Initialize without loading models (since they don't exist yet)
        system = TAMOFoA()
        print("✅ TAMO-FoA system initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize TAMO-FoA system: {e}")
        return False

def test_data_processing():
    """Test data processing utilities."""
    print("\nTesting data processing...")
    
    try:
        from utils import (
            preprocess_metrics, preprocess_logs, preprocess_traces,
            validate_data_format, DataValidator
        )
        
        # Test metrics preprocessing
        metrics = [1.0, 2.0, 3.0, 4.0, 5.0]
        processed_metrics = preprocess_metrics(metrics)
        print("✅ Metrics preprocessing works")
        
        # Test logs preprocessing
        logs = ["Error message", "Warning message", "Info message"]
        processed_logs = preprocess_logs(logs)
        print("✅ Logs preprocessing works")
        
        # Test traces preprocessing
        traces = [{"nodes": [], "edges": []}]
        processed_traces = preprocess_traces(traces)
        print("✅ Traces preprocessing works")
        
        # Test data validation
        validator = DataValidator()
        test_data = {
            'metrics': [1, 2, 3],
            'logs': ["test log"],
            'traces': {"nodes": [], "edges": []}
        }
        is_valid, errors = validator.validate_telemetry_data(test_data)
        print("✅ Data validation works")
        
        return True
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 TAMO-FoA System Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_encoder,
        test_sop_pruner,
        test_hdm2_detector,
        test_main_system,
        test_data_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! TAMO-FoA system is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
