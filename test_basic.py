#!/usr/bin/env python3
"""
Basic TAMO-FoA Test (No Dependencies)

This script performs basic tests without requiring external dependencies.
"""

import sys
import os
from pathlib import Path

def test_file_structure():
    """Test that all required files exist."""
    print("Testing file structure...")
    
    required_files = [
        "src/encoder.py",
        "src/sop_pruner.py", 
        "src/hdm2_detector.py",
        "src/main.py",
        "src/utils.py",
        "README.md",
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml",
        "LICENSE",
        ".gitignore",
        "configs/default_config.yaml",
        "experiments/run_evaluation.py",
        "scripts/setup_environment.sh"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = Path(file_path)
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def test_syntax():
    """Test Python syntax of core files."""
    print("\nTesting Python syntax...")
    
    python_files = [
        "src/encoder.py",
        "src/sop_pruner.py",
        "src/hdm2_detector.py", 
        "src/utils.py",
        "src/main.py",
        "experiments/run_evaluation.py",
        "test_system.py"
    ]
    
    syntax_errors = []
    for file_path in python_files:
        full_path = Path(file_path)
        if full_path.exists():
            try:
                with open(full_path, 'r') as f:
                    compile(f.read(), str(full_path), 'exec')
                print(f"✅ {file_path}")
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}: {e}")
                print(f"❌ {file_path}: {e}")
    
    if syntax_errors:
        print(f"\n❌ Syntax errors found: {syntax_errors}")
        return False
    
    print("✅ All Python files have valid syntax")
    return True

def test_config_format():
    """Test configuration file format."""
    print("\nTesting configuration files...")
    
    config_files = [
        "configs/default_config.yaml"
    ]
    
    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    content = f.read()
                    # Basic YAML structure check - look for key: value pairs
                    lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
                    has_yaml_structure = any(':' in line for line in lines)
                    
                    if has_yaml_structure and len(content) > 100:  # Reasonable size check
                        print(f"✅ {config_file}")
                    else:
                        print(f"❌ {config_file}: Invalid YAML format")
                        return False
            except Exception as e:
                print(f"❌ {config_file}: {e}")
                return False
    
    print("✅ Configuration files are valid")
    return True

def test_docker_files():
    """Test Docker configuration files."""
    print("\nTesting Docker files...")
    
    docker_files = ["Dockerfile", "docker-compose.yml"]
    
    for docker_file in docker_files:
        docker_path = Path(docker_file)
        if docker_path.exists():
            with open(docker_path, 'r') as f:
                content = f.read()
                if 'FROM' in content or 'version:' in content:
                    print(f"✅ {docker_file}")
                else:
                    print(f"❌ {docker_file}: Invalid Docker format")
                    return False
    
    print("✅ Docker files are valid")
    return True

def test_readme():
    """Test README content."""
    print("\nTesting README...")
    
    readme_path = Path("README.md")
    if readme_path.exists():
        with open(readme_path, 'r') as f:
            content = f.read()
            
        required_sections = [
            "TAMO-FoA",
            "Installation",
            "Usage",
            "Features"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section.lower() not in content.lower():
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ README missing sections: {missing_sections}")
            return False
        
        print("✅ README contains required sections")
        return True
    else:
        print("❌ README.md not found")
        return False

def test_requirements():
    """Test requirements.txt format."""
    print("\nTesting requirements.txt...")
    
    req_path = Path("requirements.txt")
    if req_path.exists():
        with open(req_path, 'r') as f:
            content = f.read()
            
        # Check for basic package format
        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
        
        if len(lines) > 0:
            print(f"✅ requirements.txt has {len(lines)} packages")
            return True
        else:
            print("❌ requirements.txt is empty")
            return False
    else:
        print("❌ requirements.txt not found")
        return False

def main():
    """Run all basic tests."""
    print("🧪 TAMO-FoA Basic Tests (No Dependencies)")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_syntax,
        test_config_format,
        test_docker_files,
        test_readme,
        test_requirements
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
    print(f"Basic Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! TAMO-FoA structure is correct.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full tests: python test_system.py")
        print("3. Start services: docker-compose up -d")
        return 0
    else:
        print("⚠️  Some basic tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
