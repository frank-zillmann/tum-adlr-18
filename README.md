# tum-adlr-18
**Frank Zillmann's Advanced Deep Learning for Robotics Project: Efficient Environment Exploration and 3D Reconstruction with Reinforcement Learning and Multiple View Geometry**

## Installation

1. Clone the repository:
```bash
git clone https://github.com/frank-zillmann/tum-adlr-18 --recursive
cd tum-adlr-18
```

2. Run the setup script:
```bash
source ./install/setup.sh
```

### Known fixes:
If you encounter errors with EGL (NVIDIA’s GPU offscreen rendering backend), install the following packages:
```bash
sudo apt update
sudo apt install -y libegl1-mesa-dev libgles2-mesa-dev
```

## Usage

### Testing

Test the environment setup:
```bash
python tests/test_reconstruct3D_env.py
```
