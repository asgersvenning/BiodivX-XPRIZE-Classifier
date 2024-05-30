#!/bin/bash
# Preliminary installation script for the pipeline dependencies and environment

# Static installation options
ENV_NAME="xprize_pipeline"
REQUIRED_PYTHON_VERSION="3.11"
UBUNTU_VERSION="20.04"
CUDA_VERSION="12.2"
NVIDIA_DRIVER_VERSION="450.51"

# Function to check OS and CUDA version
check_system_requirements() {
    # Check OS
    if [[ $(lsb_release -rs) != "$UBUNTU_VERSION" ]]; then
        echo "Warning: This script is designed for Ubuntu $UBUNTU_VERSION."
    fi

    # Check CUDA version
    if ! nvcc --version | grep "release $CUDA_VERSION" > /dev/null; then
        echo "Warning: This script is designed for CUDA $CUDA_VERSION."
    fi

    # Check NVIDIA driver version
    if ! nvidia-smi | grep "Driver Version: $NVIDIA_DRIVER_VERSION" > /dev/null; then
        echo "Warning: This script is designed for NVIDIA driver version $NVIDIA_DRIVER_VERSION."
    fi
}

if ! command -v micromamba &> /dev/null; then
    echo "micromamba could not be found. Please install micromamba before running this script."
    exit 1
fi

# Check system requirements
check_system_requirements

# Prepare environment
# Check if the environment exists
if micromamba env list | grep -q "^$ENV_NAME$"; then
    echo "Environment $ENV_NAME already exists. Skipping creation."
else
    echo "Creating environment $ENV_NAME with Python $REQUIRED_PYTHON_VERSION."
    micromamba create --name $ENV_NAME python="$REQUIRED_PYTHON_VERSION" -y
fi
micromamba run -n "$ENV_NAME" bash << EOF

PYTHON_VERSION=\$(python --version 2>&1 | awk '{print \$2}')
if [[ "\$PYTHON_VERSION" != "\$REQUIRED_PYTHON_VERSION"* ]]; then
    echo "Warning: The environment $ENV_NAME is using Python \$PYTHON_VERSION, but this script requires Python \$REQUIRED_PYTHON_VERSION."
fi

# Install torch; assumes the system is using CUDA>=12.1
micromamba install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge

CWD=$(pwd)

# Install flat-bug; i.e. Multiple Object Detection and Segmentation
cd "$HOME"
if [ -d "flat-bug" ]; then
  git clone git@github.com:darsa-group/flat-bug.git
fi
cd flat-bug
git fetch
git checkout dev_experiments
git pull
pip install -e .

# Install Clustering module
cd "$HOME"
if [ -d "flat-bug-clustering" ]; then
  git clone git@github.com:GuillaumeMougeot/flat-bug-clustering.git
fi
cd flat-bug-clustering
pip install -e .
cd "$HOME"

# Install Classification module
cd "$HOME"
if [ -d "xprize-classifier" ]; then
  git clone git@github.com:GuillaumeMougeot/xprize-classifier
fi
cd xprize-classifier
pip install -e .

# Return to original directory
cd "$CWD"

EOF
