#!/bin/bash

# Create conda environment
echo "Creating conda environment 'adlr'..."
conda env create -f ./install/environment.yml

# Activate the environment
conda activate adlr

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r ./install/requirements.txt

echo ""
echo "Finished! Use the environment with:"
echo "  conda activate adlr"
