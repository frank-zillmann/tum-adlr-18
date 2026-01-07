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
I used a CPU-only laptop and a Google Cloud VM with NVIDIA T4 GPU and `pytorch-2-7-cu128-ubuntu‑2404‑nvidia‑570` image. The following fixes were necessary on the VM.

**OpenCV/libGL error** (`libGL.so.1: cannot open shared object file`):
```bash
sudo apt install -y libgl1
```

**EGL error** (NVIDIA GPU offscreen rendering backend):
```bash
# Mesa EGL libraries (base requirement)
sudo apt install -y libegl1-mesa-dev libgles2-mesa-dev

# NVIDIA EGL (required for headless GPU rendering with NVIDIA drivers)
# Replace 570 with your driver version (check with: nvidia-smi)
sudo apt install -y libnvidia-gl-570-server
```

**MuJoCo rendering backend** (set EGL for headless/offscreen rendering):
```bash
export MUJOCO_GL=egl
# Add to ~/.bashrc for persistence:
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
```

**EGL device enumeration fails with Mesa** (`Cannot initialize a EGL device display`):
If you have both NVIDIA and Mesa EGL drivers installed, the device enumeration may fail on Mesa devices. Force NVIDIA-only EGL:
```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
# Add to ~/.bashrc for persistence:
echo 'export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json' >> ~/.bashrc
```

**GPU render device permission denied** (`failed to open /dev/dri/renderD128: Permission denied`):
```bash
sudo usermod -aG render $USER
sudo usermod -aG video $USER
# Log out and back in, or use newgrp to apply immediately:
newgrp render
newgrp video
```

## Usage

### Testing

