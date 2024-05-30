#!/bin/bash
# Preliminary installation script for the pipeline dependencies and environment

if ! command -v micromamba &> /dev/null; then
    echo "micromamba could not be found. Please install micromamba before running this script."
    exit 1
fi

# Prepare environment
micromamba create --name fbc
micromamba run -n <ENV_NAME> bash << EOF

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
