# TAMO-FoA Implementation Summary

## Overview
This repository contains the complete implementation of the TAMO-FoA (Tool-Augmented Multimodal Flow-of-Action) framework as described in the paper "Enhancing Cloud-Native Root Cause Analysis with Large Language Models: Recent Advances, Challenges, and the Path Towards AgentOps".

## Repository Structure
```
TAMO-FoA/
├── src/                          # Core implementation
│   ├── encoder.py               # Multi-modal encoder with diffusion fusion
│   ├── sop_pruner.py            # SOP-guided action pruner
│   ├── hdm2_detector.py         # Hallucination detection module
│   ├── main.py                  # Main TAMO-FoA framework
│   └── utils.py                 # Utility functions
├── notebooks/                    # Training notebooks
│   └── 01_encoder_training.ipynb
├── experiments/                  # Evaluation scripts
│   └── run_evaluation.py
├── configs/                      # Configuration files
│   └── default_config.yaml
├── scripts/                      # Setup and utility scripts
│   └── setup_environment.sh
├── data/                         # Data directories
│   ├── OpenRCA/
│   ├── ITBench/
│   ├── AIOpsLab/
│   └── CloudDiagBench/
├── models/                       # Model storage
│   ├── encoder/
│   ├── sop_pruner/
│   └── hdm2/
├── Dockerfile                    # Container configuration
├── docker-compose.yml           # Multi-service deployment
├── requirements.txt              # Python dependencies
├── README.md                     # Main documentation
├── LICENSE                       # Apache 2.0 license
├── .gitignore                   # Git ignore rules
├── test_system.py               # System tests
└── IMPLEMENTATION_SUMMARY.md    # This file
```

## Key Components

### 1. Multi-Modal Encoder (`src/encoder.py`)
- **Hybrid Architecture**: Combines modality-specific encoders
  - 1D-CNN for metrics (time-series)
  - BERT+LSTM for logs (textual)
  - GNN for traces (graph-structured)
- **Diffusion-Based Fusion**: Uses diffusion models for multi-modal fusion
- **Configuration**: 1024-dimensional output, 1000 diffusion steps
- **Training**: Adam optimizer, learning rate 1e-4, batch size 32

### 2. SOP-Enhanced Reasoning Agent (`src/sop_pruner.py`)
- **Knowledge Base**: Neo4j-based SOP storage
- **Retrieval**: Sentence-BERT with 0.89 recall@5
- **Action Pruning**: Random Forest classifier (94.3% accuracy)
- **Paradigm**: Thought-ActionSet-Action-Observation
- **Performance**: 42% reduction in irrelevant tool calls

### 3. HDM-2 Hallucination Detector (`src/hdm2_detector.py`)
- **Dual Verification**: External DeBERTa + internal probing
- **Contrastive Learning**: 21% improvement over baseline
- **Enterprise-Aware**: Distinguishes public vs. proprietary knowledge
- **Performance**: F1=0.82, 7.2% false positive rate

### 4. Main Framework (`src/main.py`)
- **Integration**: Combines all components
- **Performance**: 3.7s response time, 27 incidents/minute
- **Reproducibility**: Complete source code and model weights
- **Evaluation**: Comprehensive benchmarking on multiple datasets

## Performance Metrics (from Paper)

### Accuracy Results
- **OpenRCA**: 78.4% (vs. 61.2% for vanilla ReAct)
- **ITBench SRE**: 64.7% (vs. 35.5% for Flow-of-Action)
- **Precision**: 0.76
- **Recall**: 0.81

### Efficiency Improvements
- **Token Reduction**: 2.3× compared to GPT-4o-only ReAct
- **Response Time**: 3.7 seconds average
- **Throughput**: 27 incidents per minute
- **Memory Usage**: 8.2GB peak during processing

### Ablation Study Results
- **Without Diffusion Encoder**: -9.1% accuracy
- **Without SOP Guidance**: -11.7% accuracy
- **Without HDM-2 Verification**: +15.3% hallucination rate

## Installation and Usage

### Quick Start
```bash
# Clone repository
git clone https://github.com/akshaymittal143/TAMO-FoA.git
cd TAMO-FoA

# Setup environment
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# Start services
docker-compose up -d

# Run evaluation
python experiments/run_evaluation.py
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build

# Individual service commands
docker-compose up neo4j redis kafka elasticsearch prometheus grafana
```

## Evaluation Benchmarks

### Supported Datasets
1. **OpenRCA**: 335 real-world failures, 68GB telemetry
2. **ITBench**: 94 SRE scenarios
3. **AIOpsLab**: 48 end-to-end incident scenarios
4. **CloudDiagBench**: 156 multi-cloud failure scenarios

### Baseline Comparisons
- Rule-Based RCA
- Random Forest Classifier
- GPT-4o Direct Prompting
- GPT-4o with RAG
- Vanilla ReAct Agent
- TAMO-Original
- CoT-Aug (2025)
- Flow-of-Action

## Reproducibility

### Available Artifacts
- Complete source code
- Pre-trained model weights
- SOP knowledge base (247 procedures)
- Evaluation scripts
- Exact prompts and tool APIs
- Seed controls for reproducible results
- Sanitized SOP dataset

### Repository Features
- Comprehensive documentation
- Docker containerization
- Automated setup scripts
- Jupyter training notebooks
- Performance monitoring
- System testing framework

## Research Contributions

### Technical Contributions
1. **TAMO-FoA Framework**: End-to-end RCA system with domain-specific reasoning
2. **Multi-Modal Fusion Engine**: Diffusion-based fusion for heterogeneous telemetry
3. **Enterprise-Grade Verification**: HDM-2 hallucination detector
4. **Reproducible Evaluation**: Comprehensive benchmarking methodology

### Novel Aspects
- First framework to combine diffusion-based fusion with SOP-guided reasoning
- Enterprise-specific hallucination detection for AIOps
- Systematic evaluation on multiple real-world benchmarks
- Open-source implementation with full reproducibility

## Future Directions

### Planned Extensions
1. **Live ICS/OT Integration**: Industrial control system telemetry
2. **Compliance Automation**: NERC CIP, CIS Benchmarks, PCI-DSS
3. **Adversarial Robustness**: Defense against telemetry poisoning
4. **Cost-Effective Deployment**: Model distillation techniques
5. **Continuous Learning**: AlphaMLOps integration
6. **Graph-Based Context**: LongMem architectures for extended dependencies

## Citation

```bibtex
@inproceedings{mittal2026tamo,
  title={TAMO-FoA: Tool-Augmented AIOps for Reliable RCA in Cloud-Native Systems},
  author={Mittal, Akshay and Kandi, Krishna and Syed, Rafiuddin and Mahajan, Sagar},
  booktitle={Proceedings of the IEEE CCWC 2026},
  year={2026},
  organization={IEEE}
}
```

## Contact

### Authors
- **Akshay Mittal** (PhD Scholar, University of the Cumberlands): akshay.mittal@ieee.org
- **Krishna Kandi** (Independent Researcher): krishna.kandi@ieee.org
- **Rafiuddin Syed** (Independent Researcher): rafiuddinsyed01@gmail.com
- **Sagar Mahajan** (Independent Researcher): mahajanspm@yahoo.com

### Links
- **Repository**: https://github.com/akshaymittal143/TAMO-FoA
- **Paper**: IEEE CCWC 2026

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This implementation represents the complete TAMO-FoA framework as described in the research paper. All components are designed to work together to provide reliable, enterprise-grade root cause analysis for cloud-native systems.
