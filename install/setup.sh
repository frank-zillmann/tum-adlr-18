#!/bin/bash
set -e

# Get environment name
if [ -z "$1" ]; then
    read -p "Enter environment name (default: adlr): " ENV_NAME
    ENV_NAME=${ENV_NAME:-adlr}
else
    ENV_NAME=$1
fi

# Create environment
echo "Step 1/4: Creating and activating environment $ENV_NAME with Python 3.10..."
conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

echo ""
# Detect CUDA
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "CUDA detected - installing PyTorch with CUDA 12.8 support"
    PYTORCH_PACKAGES="pytorch torchvision pytorch-cuda=12.8 -c pytorch -c nvidia"
else
    echo "No CUDA detected - installing CPU-only PyTorch"
    PYTORCH_PACKAGES="pytorch torchvision cpuonly -c pytorch"
fi
# Install PyTorch
echo ""
echo "Step 2/4: Installing PyTorch packages: $PYTORCH_PACKAGES ..."
conda install $PYTORCH_PACKAGES -y

# Install remaining packages from yml
echo ""
echo "Step 3/4: Installing remaining packages from environment.yml..."
conda env update --name $ENV_NAME --file install/environment.yml --prune

# Install repository in editable mode
echo ""
echo "Step 4/4: Installing repository packages in editable mode..."
pip install -e external/robosuite
pip install -e .

echo ""
echo "Installation and activation of '$ENV_NAME' completed!"
