#!/bin/bash

# TAMO-FoA Environment Setup Script
# This script sets up the complete TAMO-FoA environment

set -e

echo "🚀 Setting up TAMO-FoA Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.8+ is installed
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.8+ is required but not installed"
        exit 1
    fi
}

# Check if pip is installed
check_pip() {
    print_status "Checking pip installation..."
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
    else
        print_error "pip3 is required but not installed"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    print_success "Python dependencies installed"
}

# Create necessary directories
create_directories() {
    print_status "Creating directory structure..."
    
    directories=(
        "data/OpenRCA"
        "data/ITBench"
        "data/AIOpsLab"
        "data/CloudDiagBench"
        "models/encoder"
        "models/sop_pruner"
        "models/hdm2"
        "logs"
        "cache"
        "results"
        "configs/grafana/provisioning"
        "configs/grafana/dashboards"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    print_success "Directory structure created"
}

# Check Docker installation
check_docker() {
    print_status "Checking Docker installation..."
    if command -v docker &> /dev/null; then
        print_success "Docker found"
        if command -v docker-compose &> /dev/null; then
            print_success "Docker Compose found"
        else
            print_warning "Docker Compose not found. Some features may not work."
        fi
    else
        print_warning "Docker not found. You can still run TAMO-FoA without containerized services."
    fi
}

# Initialize Neo4j with sample data
init_neo4j() {
    print_status "Initializing Neo4j with sample SOPs..."
    
    # Create sample SOP data
    cat > data/sample_sops.json << EOF
[
  {
    "sop_id": "sop_001",
    "title": "Database Connection Pool Issues",
    "description": "Troubleshooting database connection pool problems",
    "steps": [
      "Check connection pool metrics",
      "Monitor active connections",
      "Review application configuration",
      "Restart service if necessary"
    ],
    "category": "database",
    "tags": ["database", "connections", "pool"]
  },
  {
    "sop_id": "sop_002",
    "title": "High CPU Usage Investigation",
    "description": "Investigate and resolve high CPU usage",
    "steps": [
      "Check CPU usage metrics",
      "Identify top processes",
      "Analyze system logs",
      "Scale resources if needed"
    ],
    "category": "performance",
    "tags": ["cpu", "performance", "scaling"]
  },
  {
    "sop_id": "sop_003",
    "title": "Memory Leak Detection",
    "description": "Detect and resolve memory leaks",
    "steps": [
      "Monitor memory usage trends",
      "Check for memory leaks in logs",
      "Analyze heap dumps",
      "Restart affected services"
    ],
    "category": "memory",
    "tags": ["memory", "leak", "heap"]
  }
]
EOF
    
    print_success "Sample SOP data created"
}

# Create sample configuration files
create_configs() {
    print_status "Creating configuration files..."
    
    # Prometheus configuration
    cat > configs/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'tamo-foa'
    static_configs:
      - targets: ['tamo-foa:8000']
    scrape_interval: 5s
EOF

    # Grafana datasource configuration
    mkdir -p configs/grafana/provisioning/datasources
    cat > configs/grafana/provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    print_success "Configuration files created"
}

# Run tests
run_tests() {
    print_status "Running basic tests..."
    
    # Test Python imports
    python3 -c "
import sys
sys.path.append('src')
try:
    from encoder import MultiModalDiffusionEncoder
    from sop_pruner import SOPPruner
    from hdm2_detector import HDM2Detector
    from utils import SystemConfig
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"
    
    print_success "Basic tests passed"
}

# Main setup function
main() {
    echo "=================================="
    echo "    TAMO-FoA Environment Setup    "
    echo "=================================="
    
    # Check prerequisites
    check_python
    check_pip
    check_docker
    
    # Setup Python environment
    create_venv
    activate_venv
    install_dependencies
    
    # Setup directories and configs
    create_directories
    init_neo4j
    create_configs
    
    # Run tests
    run_tests
    
    echo ""
    echo "=================================="
    print_success "TAMO-FoA Environment Setup Complete!"
    echo "=================================="
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Start services with Docker: docker-compose up -d"
    echo "3. Run training notebooks in the notebooks/ directory"
    echo "4. Execute evaluation: python experiments/run_evaluation.py"
    echo ""
    echo "For more information, see README.md"
}

# Run main function
main "$@"
