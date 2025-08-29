#!/bin/bash

# SLAM-LLM Installation Script
# Following README.md installation instructions

set -e

echo "Starting SLAM-LLM installation..."

# Activate virtual environment
source slam_env/bin/activate

# Install transformers v4.35.2
echo "Installing transformers v4.35.2..."
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout tags/v4.35.2
pip install -e .
cd ..

# Install peft v0.6.0
echo "Installing peft v0.6.0..."
git clone https://github.com/huggingface/peft.git
cd peft
git checkout tags/v0.6.0
pip install -e .
cd ..

# Install PyTorch with CUDA 11.8
echo "Installing PyTorch 2.0.1 with CUDA 11.8..."
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Install SLAM-LLM
echo "Installing SLAM-LLM..."
pip install -e .

# Install fairseq (optional)
echo "Installing fairseq..."
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ..

echo "SLAM-LLM installation completed!"
echo "To activate the environment, run: source slam_env/bin/activate"