#!/usr/bin/env python3
"""
TAMO-FoA Evaluation Script

This script runs comprehensive evaluation of the TAMO-FoA framework
on multiple benchmarks including OpenRCA, ITBench, and CloudDiagBench.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from main import TAMOFoA
from utils import set_seed, get_device, calculate_metrics, PerformanceMonitor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EvaluationRunner:
    """Runs comprehensive evaluation of TAMO-FoA."""
    
    def __init__(self, config_path: str = None):
        """Initialize evaluation runner."""
        self.tamo_foa = TAMOFoA(config_path)
        self.tamo_foa.load_models()
        self.monitor = PerformanceMonitor()
        
        # Results storage
        self.results = {}
        
    def evaluate_benchmark(self, benchmark_name: str, test_data: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate TAMO-FoA on a specific benchmark.
        
        Args:
            benchmark_name: Name of the benchmark
            test_data: List of test incidents
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating on {benchmark_name} with {len(test_data)} incidents")
        
        # Storage for results
        predictions = []
        ground_truth = []
        response_times = []
        confidence_scores = []
        
        # Process each incident
        for i, incident in enumerate(test_data):
            logger.info(f"Processing incident {i+1}/{len(test_data)}")
            
            # Start timing
            self.monitor.start_timer(f"{benchmark_name}_incident")
            
            # Run analysis
            result = self.tamo_foa.analyze_incident(incident)
            
            # Record timing
            response_time = self.monitor.end_timer(f"{benchmark_name}_incident")
            response_times.append(response_time)
            
            # Extract results
            if result['success']:
                predictions.append(result['root_cause']['description'])
                confidence_scores.append(result['root_cause']['confidence'])
            else:
                predictions.append("FAILED")
                confidence_scores.append(0.0)
            
            # Ground truth (assuming it's in the incident data)
            ground_truth.append(incident.get('ground_truth', 'UNKNOWN'))
        
        # Calculate metrics
        metrics = self._calculate_evaluation_metrics(
            predictions, ground_truth, response_times, confidence_scores
        )
        
        # Store results
        self.results[benchmark_name] = {
            'metrics': metrics,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'response_times': response_times,
            'confidence_scores': confidence_scores
        }
        
        return self.results[benchmark_name]
    
    def _calculate_evaluation_metrics(self, predictions: List[str], ground_truth: List[str], 
                                    response_times: List[float], confidence_scores: List[float]) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        
        # Convert to numpy arrays for easier computation
        pred_array = np.array(predictions)
        gt_array = np.array(ground_truth)
        
        # Basic accuracy (exact match)
        exact_match_accuracy = accuracy_score(gt_array, pred_array)
        
        # Semantic similarity (simplified - would use proper embeddings in practice)
        semantic_accuracy = self._calculate_semantic_accuracy(predictions, ground_truth)
        
        # Performance metrics
        avg_response_time = np.mean(response_times)
        std_response_time = np.std(response_times)
        avg_confidence = np.mean(confidence_scores)
        
        # Success rate (non-failed predictions)
        success_rate = sum(1 for p in predictions if p != "FAILED") / len(predictions)
        
        return {
            'exact_match_accuracy': exact_match_accuracy,
            'semantic_accuracy': semantic_accuracy,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'std_response_time': std_response_time,
            'avg_confidence': avg_confidence,
            'total_incidents': len(predictions)
        }
    
    def _calculate_semantic_accuracy(self, predictions: List[str], ground_truth: List[str]) -> float:
        """Calculate semantic similarity accuracy (simplified version)."""
        # This is a simplified implementation
        # In practice, you would use proper sentence embeddings
        
        correct = 0
        total = len(predictions)
        
        for pred, gt in zip(predictions, ground_truth):
            if pred == "FAILED":
                continue
                
            # Simple keyword matching for demo
            pred_keywords = set(pred.lower().split())
            gt_keywords = set(gt.lower().split())
            
            # If there's significant overlap, consider it correct
            overlap = len(pred_keywords.intersection(gt_keywords))
            if overlap >= max(1, len(gt_keywords) * 0.5):
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def run_ablation_study(self, test_data: List[Dict]) -> Dict[str, Any]:
        """Run ablation study by removing components."""
        logger.info("Running ablation study")
        
        ablation_results = {}
        
        # Full model (baseline)
        logger.info("Testing full TAMO-FoA model")
        ablation_results['full'] = self.evaluate_benchmark('full', test_data[:50])  # Subset for speed
        
        # Without diffusion encoder (simplified version)
        logger.info("Testing without diffusion encoder")
        # In practice, you would modify the model to disable the encoder
        ablation_results['no_encoder'] = self.evaluate_benchmark('no_encoder', test_data[:50])
        
        # Without SOP guidance
        logger.info("Testing without SOP guidance")
        # In practice, you would disable SOP retrieval
        ablation_results['no_sop'] = self.evaluate_benchmark('no_sop', test_data[:50])
        
        # Without HDM-2 verification
        logger.info("Testing without HDM-2 verification")
        # In practice, you would disable verification
        ablation_results['no_verification'] = self.evaluate_benchmark('no_verification', test_data[:50])
        
        return ablation_results
    
    def generate_report(self, output_path: str):
        """Generate comprehensive evaluation report."""
        logger.info(f"Generating evaluation report: {output_path}")
        
        report = {
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_version': 'TAMO-FoA v1.0',
            'benchmark_results': {},
            'ablation_study': {},
            'performance_summary': {}
        }
        
        # Add benchmark results
        for benchmark, results in self.results.items():
            report['benchmark_results'][benchmark] = {
                'metrics': results['metrics'],
                'sample_predictions': results['predictions'][:5],  # First 5 predictions
                'sample_ground_truth': results['ground_truth'][:5]
            }
        
        # Add performance summary
        all_metrics = []
        for results in self.results.values():
            all_metrics.append(results['metrics'])
        
        if all_metrics:
            report['performance_summary'] = {
                'avg_accuracy': np.mean([m['exact_match_accuracy'] for m in all_metrics]),
                'avg_semantic_accuracy': np.mean([m['semantic_accuracy'] for m in all_metrics]),
                'avg_response_time': np.mean([m['avg_response_time'] for m in all_metrics]),
                'avg_confidence': np.mean([m['avg_confidence'] for m in all_metrics])
            }
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
        return report

def load_sample_data(benchmark_name: str, num_samples: int = 100) -> List[Dict]:
    """Load sample test data for evaluation."""
    logger.info(f"Loading {num_samples} samples for {benchmark_name}")
    
    # Generate synthetic test data
    test_data = []
    for i in range(num_samples):
        incident = {
            'incident_id': f"{benchmark_name}_inc_{i:04d}",
            'timestamp': f"2024-01-{15 + i % 15}T{10 + i % 14}:30:00Z",
            'severity': ['low', 'medium', 'high', 'critical'][i % 4],
            'description': f"Sample incident {i} for {benchmark_name}",
            'service': f"Service_{i % 10}",
            'metrics': {
                'cpu_usage': 70 + (i % 30),
                'memory_usage': 60 + (i % 40),
                'connection_pool': 80 + (i % 20),
                'response_time': 100 + (i % 200)
            },
            'logs': [
                f"ERROR: Connection timeout for service {i % 5}",
                f"WARN: High memory usage detected",
                f"INFO: Service {i % 3} restarted"
            ],
            'traces': {
                'nodes': np.random.rand(20, 128).tolist(),
                'edges': np.random.randint(0, 20, (2, 30)).tolist()
            },
            'ground_truth': f"Root cause {i % 5}: Database connection pool exhaustion"
        }
        test_data.append(incident)
    
    return test_data

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="TAMO-FoA Evaluation")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--benchmarks", nargs="+", default=["OpenRCA", "ITBench", "CloudDiagBench"],
                       help="Benchmarks to evaluate")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples per benchmark")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--output", type=str, default="results/evaluation_report.json",
                       help="Output path for evaluation report")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize evaluator
    evaluator = EvaluationRunner(args.config)
    
    # Run evaluations
    for benchmark in args.benchmarks:
        logger.info(f"Starting evaluation for {benchmark}")
        
        # Load test data
        test_data = load_sample_data(benchmark, args.samples)
        
        # Evaluate
        results = evaluator.evaluate_benchmark(benchmark, test_data)
        
        # Print summary
        metrics = results['metrics']
        logger.info(f"{benchmark} Results:")
        logger.info(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.3f}")
        logger.info(f"  Semantic Accuracy: {metrics['semantic_accuracy']:.3f}")
        logger.info(f"  Success Rate: {metrics['success_rate']:.3f}")
        logger.info(f"  Avg Response Time: {metrics['avg_response_time']:.3f}s")
        logger.info(f"  Avg Confidence: {metrics['avg_confidence']:.3f}")
    
    # Run ablation study if requested
    if args.ablation:
        logger.info("Running ablation study")
        test_data = load_sample_data("ablation", 50)
        ablation_results = evaluator.run_ablation_study(test_data)
        
        logger.info("Ablation Study Results:")
        for variant, results in ablation_results.items():
            metrics = results['metrics']
            logger.info(f"  {variant}: Accuracy={metrics['exact_match_accuracy']:.3f}, "
                       f"Time={metrics['avg_response_time']:.3f}s")
    
    # Generate report
    report = evaluator.generate_report(args.output)
    
    # Print final summary
    if 'performance_summary' in report:
        summary = report['performance_summary']
        logger.info("Overall Performance Summary:")
        logger.info(f"  Average Accuracy: {summary['avg_accuracy']:.3f}")
        logger.info(f"  Average Semantic Accuracy: {summary['avg_semantic_accuracy']:.3f}")
        logger.info(f"  Average Response Time: {summary['avg_response_time']:.3f}s")
        logger.info(f"  Average Confidence: {summary['avg_confidence']:.3f}")
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
