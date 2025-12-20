#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "============================================"
echo "ADLR Environment Setup"
echo "============================================"
echo ""

# Get environment name
if [ -z "$1" ]; then
    read -p "Enter environment name (default: adlr): " ENV_NAME
    ENV_NAME=${ENV_NAME:-adlr}
else
    ENV_NAME=$1
fi

echo "Creating environment: $ENV_NAME"
echo ""

# Detect CUDA
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✓ CUDA detected - installing PyTorch with CUDA 12.8 support"
    PYTORCH_PACKAGES="pytorch torchvision pytorch-cuda=12.8 -c pytorch -c nvidia"
else
    echo "✓ No CUDA detected - installing CPU-only PyTorch"
    PYTORCH_PACKAGES="pytorch torchvision cpuonly -c pytorch"
fi
echo ""

# Create environment and install PyTorch
echo "Step 1/2: Creating environment and installing PyTorch..."
conda create -n $ENV_NAME python=3.10 $PYTORCH_PACKAGES -y

# Install remaining packages from yml
echo ""
echo "Step 2/2: Installing remaining packages from environment.yml..."
conda env update -n $ENV_NAME -f ./install/environment.yml

echo ""
echo "============================================"
echo "✓ Installation Complete!"
echo "============================================"
echo ""
echo "Activate with: conda activate $ENV_NAME"
echo ""
