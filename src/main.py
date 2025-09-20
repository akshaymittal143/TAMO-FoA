"""
TAMO-FoA Main Entry Point

This module provides the main interface for the TAMO-FoA framework,
integrating all components for end-to-end root cause analysis.
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some features may not work.")

import asyncio
import logging
import argparse
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

# Import TAMO-FoA components
try:
    from encoder import MultiModalEncoder, EncoderConfig
    from sop_pruner import SOPPruner, SOPPrunerConfig
    from hdm2_detector import HDM2Detector, HDM2Config
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some components not available: {e}")
    COMPONENTS_AVAILABLE = False

from utils import (
    SystemConfig, set_seed, get_device, load_json, save_json,
    validate_data_format, PerformanceMonitor, DataValidator
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TAMOFoA:
    """
    Main TAMO-FoA framework class that integrates all components.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize TAMO-FoA framework.
        
        Args:
            config_path: Path to system configuration file
        """
        # Load configuration
        if config_path and Path(config_path).exists():
            self.config = SystemConfig.load(config_path)
        else:
            self.config = SystemConfig()
        
        # Set device and seed
        self.device = get_device()
        set_seed(42)
        
        # Initialize components
        self.encoder = None
        self.sop_pruner = None
        self.hdm2_detector = None
        
        # Performance monitoring
        self.monitor = PerformanceMonitor()
        
        # Data validator
        self.validator = DataValidator()
        
        logger.info("TAMO-FoA framework initialized")
    
    def load_models(self):
        """Load pre-trained models for all components."""
        try:
            # Load encoder
            logger.info("Loading multi-modal encoder...")
            self.encoder = MultiModalEncoder.load(self.config.encoder_model_path)
            self.encoder.to(self.device)
            self.encoder.eval()
            
            # Load SOP pruner
            logger.info("Loading SOP pruner...")
            self.sop_pruner = SOPPruner.load(self.config.sop_pruner_path)
            
            # Load HDM-2 detector
            logger.info("Loading HDM-2 detector...")
            self.hdm2_detector = HDM2Detector.load(self.config.hdm2_model_path)
            self.hdm2_detector.to(self.device)
            self.hdm2_detector.eval()
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def analyze_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform root cause analysis on an incident.
        
        Args:
            incident_data: Incident data containing metrics, logs, and traces
            
        Returns:
            Analysis results with root cause and confidence
        """
        # Validate input data
        is_valid, errors = self.validator.validate_telemetry_data(incident_data)
        if not is_valid:
            return {
                'success': False,
                'error': f"Invalid input data: {errors}",
                'root_cause': None,
                'confidence': 0.0
            }
        
        try:
            # Step 1: Multi-modal encoding
            self.monitor.start_timer("encoding")
            
            metrics = torch.tensor(incident_data['metrics'], dtype=torch.float32).to(self.device)
            logs = incident_data['logs']
            traces_data = incident_data['traces']
            
            # Encode multi-modal data
            encoded_features = self.encoder(metrics, logs, traces_data)
            
            encoding_time = self.monitor.end_timer("encoding")
            logger.info(f"Multi-modal encoding completed in {encoding_time:.3f}s")
            
            # Step 2: SOP-guided reasoning
            self.monitor.start_timer("sop_reasoning")
            
            # Create incident context
            incident_context = self._create_incident_context(incident_data)
            
            # Get relevant SOPs
            relevant_sops = self.sop_pruner.sop_kb.retrieve_relevant_sops(
                incident_context, top_k=5
            )
            
            # Generate candidate actions
            candidate_actions = self._generate_candidate_actions(incident_data, relevant_sops)
            
            # Prune actions using SOP guidance
            relevant_actions = self.sop_pruner.prune_actions(candidate_actions, incident_context)
            
            sop_reasoning_time = self.monitor.end_timer("sop_reasoning")
            logger.info(f"SOP-guided reasoning completed in {sop_reasoning_time:.3f}s")
            
            # Step 3: Root cause analysis
            self.monitor.start_timer("rca_analysis")
            
            # Perform RCA using relevant actions
            root_cause = self._perform_rca(incident_data, relevant_actions, encoded_features)
            
            rca_time = self.monitor.end_timer("rca_analysis")
            logger.info(f"Root cause analysis completed in {rca_time:.3f}s")
            
            # Step 4: Hallucination detection and verification
            self.monitor.start_timer("verification")
            
            # Verify the root cause analysis
            verification_result = self.hdm2_detector.verify(
                root_cause['description'],
                incident_context,
                method="all"
            )
            
            verification_time = self.monitor.end_timer("verification")
            logger.info(f"Verification completed in {verification_time:.3f}s")
            
            # Compile results
            results = {
                'success': True,
                'root_cause': {
                    'description': root_cause['description'],
                    'confidence': verification_result['confidence'],
                    'evidence': root_cause['evidence'],
                    'suggested_actions': relevant_actions[:3]  # Top 3 actions
                },
                'verification': {
                    'is_factual': verification_result['is_factual'],
                    'context_score': verification_result['context_score'],
                    'knowledge_score': verification_result['knowledge_score'],
                    'consistency_score': verification_result['consistency_score']
                },
                'performance': {
                    'total_time': encoding_time + sop_reasoning_time + rca_time + verification_time,
                    'encoding_time': encoding_time,
                    'sop_reasoning_time': sop_reasoning_time,
                    'rca_time': rca_time,
                    'verification_time': verification_time
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during incident analysis: {e}")
            return {
                'success': False,
                'error': str(e),
                'root_cause': None,
                'confidence': 0.0
            }
    
    def _create_incident_context(self, incident_data: Dict[str, Any]) -> str:
        """Create context string for incident analysis."""
        context_parts = []
        
        # Add basic incident information
        if 'incident_id' in incident_data:
            context_parts.append(f"Incident ID: {incident_data['incident_id']}")
        
        if 'description' in incident_data:
            context_parts.append(f"Description: {incident_data['description']}")
        
        if 'severity' in incident_data:
            context_parts.append(f"Severity: {incident_data['severity']}")
        
        # Add metrics summary
        if 'metrics' in incident_data:
            metrics = incident_data['metrics']
            if isinstance(metrics, dict):
                metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
                context_parts.append(f"Metrics: {metrics_str}")
        
        # Add log summary
        if 'logs' in incident_data and incident_data['logs']:
            log_count = len(incident_data['logs'])
            error_logs = [log for log in incident_data['logs'] if 'error' in log.lower()]
            context_parts.append(f"Logs: {log_count} total, {len(error_logs)} errors")
        
        return ". ".join(context_parts)
    
    def _generate_candidate_actions(self, incident_data: Dict[str, Any], 
                                   relevant_sops: List[Dict]) -> List[Dict]:
        """Generate candidate actions based on incident data and SOPs."""
        candidate_actions = []
        
        # Generate actions from SOPs
        for sop in relevant_sops:
            for step in sop.get('steps', []):
                action = self._extract_action_from_sop_step(step, sop)
                if action:
                    candidate_actions.append(action)
        
        # Generate actions based on incident type
        if 'metrics' in incident_data:
            metrics = incident_data['metrics']
            if isinstance(metrics, dict):
                # High CPU usage
                if metrics.get('cpu_usage', 0) > 80:
                    candidate_actions.append({
                        'type': 'query_metrics',
                        'description': 'Check CPU usage details',
                        'parameters': {'metric': 'cpu_usage'},
                        'priority': 0.8
                    })
                
                # High memory usage
                if metrics.get('memory_usage', 0) > 80:
                    candidate_actions.append({
                        'type': 'query_metrics',
                        'description': 'Check memory usage details',
                        'parameters': {'metric': 'memory_usage'},
                        'priority': 0.8
                    })
                
                # Connection pool issues
                if metrics.get('connection_pool', 0) > 90:
                    candidate_actions.append({
                        'type': 'query_logs',
                        'description': 'Check connection pool logs',
                        'parameters': {'service': 'database'},
                        'priority': 0.9
                    })
        
        return candidate_actions
    
    def _extract_action_from_sop_step(self, step: str, sop: Dict) -> Optional[Dict]:
        """Extract action from SOP step description."""
        step_lower = step.lower()
        
        if 'check' in step_lower or 'query' in step_lower:
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
            'priority': 0.8
        }
    
    def _perform_rca(self, incident_data: Dict[str, Any], 
                    relevant_actions: List[Dict], 
                    encoded_features: torch.Tensor) -> Dict[str, Any]:
        """Perform root cause analysis using relevant actions."""
        # This is a simplified RCA implementation
        # In a real system, this would involve more sophisticated reasoning
        
        evidence = []
        confidence = 0.0
        
        # Analyze metrics
        if 'metrics' in incident_data:
            metrics = incident_data['metrics']
            if isinstance(metrics, dict):
                if metrics.get('connection_pool', 0) > 90:
                    evidence.append("Connection pool utilization > 90%")
                    confidence += 0.3
                
                if metrics.get('cpu_usage', 0) > 80:
                    evidence.append("High CPU usage detected")
                    confidence += 0.2
                
                if metrics.get('memory_usage', 0) > 80:
                    evidence.append("High memory usage detected")
                    confidence += 0.2
        
        # Analyze logs
        if 'logs' in incident_data:
            logs = incident_data['logs']
            error_logs = [log for log in logs if 'error' in log.lower()]
            if error_logs:
                evidence.append(f"{len(error_logs)} error logs found")
                confidence += 0.2
        
        # Generate root cause description
        if confidence > 0.5:
            if any('connection' in evidence_item.lower() for evidence_item in evidence):
                description = "Root cause: Database connection pool exhaustion"
            elif any('cpu' in evidence_item.lower() for evidence_item in evidence):
                description = "Root cause: High CPU utilization"
            elif any('memory' in evidence_item.lower() for evidence_item in evidence):
                description = "Root cause: Memory pressure"
            else:
                description = "Root cause: System resource constraints"
        else:
            description = "Root cause: Insufficient data for definitive analysis"
            confidence = 0.3
        
        return {
            'description': description,
            'evidence': evidence,
            'confidence': min(confidence, 1.0)
        }
    
    def batch_analyze(self, incidents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze multiple incidents in batch."""
        results = []
        
        for i, incident in enumerate(incidents):
            logger.info(f"Analyzing incident {i+1}/{len(incidents)}")
            result = self.analyze_incident(incident)
            results.append(result)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'encoding_stats': self.monitor.get_stats('encoding'),
            'sop_reasoning_stats': self.monitor.get_stats('sop_reasoning'),
            'rca_stats': self.monitor.get_stats('rca_analysis'),
            'verification_stats': self.monitor.get_stats('verification')
        }

def main():
    """Main entry point for TAMO-FoA."""
    parser = argparse.ArgumentParser(description="TAMO-FoA Root Cause Analysis")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--incident", type=str, help="Path to incident data file")
    parser.add_argument("--batch", type=str, help="Path to batch incidents file")
    parser.add_argument("--output", type=str, help="Path to output file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize TAMO-FoA
    tamo_foa = TAMOFoA(args.config)
    
    # Load models
    tamo_foa.load_models()
    
    if args.incident:
        # Analyze single incident
        incident_data = load_json(args.incident)
        result = tamo_foa.analyze_incident(incident_data)
        
        if args.output:
            save_json(result, args.output)
        else:
            print(json.dumps(result, indent=2))
    
    elif args.batch:
        # Analyze batch of incidents
        incidents = load_json(args.batch)
        results = tamo_foa.batch_analyze(incidents)
        
        if args.output:
            save_json(results, args.output)
        else:
            print(json.dumps(results, indent=2))
    
    else:
        # Interactive mode
        print("TAMO-FoA Interactive Mode")
        print("Enter incident data as JSON (or 'quit' to exit):")
        
        while True:
            try:
                user_input = input("> ")
                if user_input.lower() == 'quit':
                    break
                
                incident_data = json.loads(user_input)
                result = tamo_foa.analyze_incident(incident_data)
                print(json.dumps(result, indent=2))
                
            except json.JSONDecodeError:
                print("Invalid JSON. Please try again.")
            except KeyboardInterrupt:
                break
    
    # Print performance stats
    stats = tamo_foa.get_performance_stats()
    print("\nPerformance Statistics:")
    for component, stat in stats.items():
        if stat:
            print(f"{component}: {stat}")

if __name__ == "__main__":
    main()
