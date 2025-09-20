# TAMO-FoA: Tool-Augmented Multimodal Observation with Flow-of-Action

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

**TAMO-FoA** is a comprehensive framework for LLM-powered Root Cause Analysis (RCA) in cloud-native systems. It addresses the three critical limitations of LLMs in production RCA: hallucination, context constraints, and dynamic dependency misunderstanding.

## 🚀 Key Features

- **Hybrid Multi-Modal Encoder**: Diffusion-based fusion for metrics, logs, and traces
- **SOP-Guided Reasoning**: Standard Operating Procedure integration with 42% reduction in irrelevant tool calls
- **HDM-2 Hallucination Detector**: Enterprise-grade verification with 21% improvement in F1 scores
- **Comprehensive Evaluation**: Benchmarks on OpenRCA, ITBench, AIOpsLab, and CloudDiagBench

## 📊 Performance Results

- **17.2% accuracy improvement** (78.4% vs 61.2% for vanilla ReAct)
- **2.3× token reduction** through SOP-guided reasoning
- **0.82 F1 score** for hallucination detection
- **3.7s average response time** with 27 incidents/minute throughput

## 🏗️ Installation

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/<username>/TAMO-FoA.git
cd TAMO-FoA

# Build Docker image with GPU support
docker build -t tamo-foa:latest .

# Run container with GPU access
docker run --gpus all -it -v $(pwd):/workspace tamo-foa:latest
```

### Option 2: Python Virtual Environment

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📁 Repository Structure

```
TAMO-FoA/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── data/                    # Dataset storage
│   ├── OpenRCA/            # OpenRCA benchmark data
│   ├── ITBench/            # ITBench scenarios
│   ├── AIOpsLab/           # AIOpsLab incidents
│   └── CloudDiagBench/     # Multi-cloud scenarios
├── models/                  # Pre-trained model weights
│   ├── encoder/            # Diffusion encoder weights
│   ├── sop_pruner/         # Random Forest pruner
│   └── hdm2/               # HDM-2 detector weights
├── notebooks/              # Jupyter notebooks
│   ├── preprocessing.ipynb # Data preprocessing
│   ├── training_encoder.ipynb
│   ├── training_sop_pruner.ipynb
│   ├── training_hdm2.ipynb
│   └── evaluation.ipynb    # Reproduce results
├── src/                    # Core implementation
│   ├── encoder.py          # Multi-modal encoder
│   ├── sop_pruner.py       # SOP-guided reasoning
│   ├── hdm2_detector.py    # Hallucination detection
│   └── utils.py            # Utility functions
└── experiments/            # Experiment scripts
    ├── run_all.sh          # End-to-end experiments
    ├── evaluate_benchmarks.sh
    └── ablation_studies.sh
```

## 📊 Data Preparation

### 1. Download Datasets

```bash
# Download OpenRCA dataset
wget https://github.com/openrca/openrca/releases/download/v1.0/openrca_dataset.zip
unzip openrca_dataset.zip -d data/OpenRCA/

# Download ITBench scenarios
wget https://github.com/microsoft/ITBench/releases/download/v1.0/itbench_scenarios.zip
unzip itbench_scenarios.zip -d data/ITBench/

# Download AIOpsLab incidents
wget https://github.com/aiopslab/aiopslab/releases/download/v1.0/aiopslab_incidents.zip
unzip aiopslab_incidents.zip -d data/AIOpsLab/

# Download CloudDiagBench
wget https://github.com/clouddiagbench/clouddiagbench/releases/download/v1.0/clouddiagbench.zip
unzip clouddiagbench.zip -d data/CloudDiagBench/
```

### 2. Preprocess Data

```bash
# Run preprocessing notebook
jupyter notebook notebooks/preprocessing.ipynb

# Or use command line
python src/utils.py --preprocess --dataset all
```

## 🎯 Training

### 1. Train Multi-Modal Encoder

```bash
# Using notebook (recommended)
jupyter notebook notebooks/training_encoder.ipynb

# Or command line
python src/encoder.py --train \
    --data_dir data/OpenRCA \
    --model_dir models/encoder \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 100
```

### 2. Train SOP Pruner

```bash
# Using notebook
jupyter notebook notebooks/training_sop_pruner.ipynb

# Or command line
python src/sop_pruner.py --train \
    --sop_data data/sop_knowledge_base.json \
    --model_dir models/sop_pruner \
    --n_estimators 100 \
    --max_depth 10
```

### 3. Train HDM-2 Detector

```bash
# Using notebook
jupyter notebook notebooks/training_hdm2.ipynb

# Or command line
python src/hdm2_detector.py --train \
    --data_dir data/OpenRCA \
    --model_dir models/hdm2 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --epochs 10
```

## 📈 Evaluation

### Reproduce All Results

```bash
# Run complete evaluation pipeline
./experiments/run_all.sh

# Or run individual benchmarks
./experiments/evaluate_benchmarks.sh
```

### Generate Specific Results

```bash
# Reproduce Table II (Performance Comparison)
jupyter notebook notebooks/evaluation.ipynb

# Run ablation studies
./experiments/ablation_studies.sh
```

## 🔬 Usage Example

```python
from src.encoder import MultiModalEncoder
from src.sop_pruner import SOPPruner
from src.hdm2_detector import HDM2Detector

# Initialize TAMO-FoA components
encoder = MultiModalEncoder.load("models/encoder/")
sop_pruner = SOPPruner.load("models/sop_pruner/")
hdm2_detector = HDM2Detector.load("models/hdm2/")

# Process incident
incident_data = {
    "metrics": {...},  # Time-series metrics
    "logs": {...},     # Log entries
    "traces": {...}    # Distributed traces
}

# Multi-modal encoding
encoded_features = encoder.encode(incident_data)

# SOP-guided reasoning
sop_guidance = sop_pruner.get_sop_guidance(incident_data)

# Hallucination detection
verification_result = hdm2_detector.verify(response, context)

print(f"Root Cause: {response}")
print(f"Confidence: {verification_result.confidence}")
```

## 📊 Benchmark Results

| Method | OpenRCA Acc. (%) | ITBench SRE (%) | Precision | Recall |
|--------|------------------|-----------------|-----------|---------|
| RB-RCA | 42.3±2.1 | 28.1±1.8 | 0.51±0.02 | 0.39±0.02 |
| RF-RCA | 58.7±3.2 | 31.4±2.9 | 0.62±0.03 | 0.55±0.03 |
| GPT-4o-Direct | 52.1±2.8 | 13.8±1.5 | 0.48±0.02 | 0.61±0.02 |
| ReAct-Vanilla | 61.2±3.5 | 41.2±3.2 | 0.64±0.03 | 0.58±0.03 |
| **TAMO-FoA** | **78.4±4.2** | **64.7±4.0** | **0.76±0.03** | **0.81±0.03** |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 Citation

If you use TAMO-FoA in your research, please cite our paper:

```bibtex
@inproceedings{mittal2025tamo,
  title={Enhancing Cloud-Native Root Cause Analysis with Large Language Models: The TAMO-FoA Framework},
  author={Mittal, Akshay},
  booktitle={Proceedings of the IEEE International Conference on Information and Communication Technology},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenRCA team for the benchmark dataset
- Microsoft ITBench for SRE scenarios
- AIOpsLab for incident data
- CloudDiagBench for multi-cloud scenarios

## 📞 Contact

- **Akshay Mittal**: akshay.mittal@ieee.org
- **Project Link**: https://github.com/<username>/TAMO-FoA
- **Paper**: [arXiv link when available]

---

**Note**: This repository contains the complete implementation of TAMO-FoA as described in our paper. All code, data, and model weights are provided for full reproducibility.
