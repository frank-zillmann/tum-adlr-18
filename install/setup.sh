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
    #TODO: Check if it is okay to use CUDA 12.4 here also the image is pytorch-2-7-cu128-ubuntu‑2404‑nvidia‑570
    echo "CUDA detected - installing PyTorch with CUDA 12.4 support"
    PYTORCH_PACKAGES="pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia"
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
conda env update --name $ENV_NAME --file install/environment.yml --verbose

# Install repository in editable mode
echo ""
echo "Step 4/4: Installing repository packages in editable mode..."
pip install -e external/robosuite
pip install -e .

echo ""
echo "Installation and activation of '$ENV_NAME' completed!"
