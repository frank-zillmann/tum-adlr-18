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

## Usage
### Training
To train a model, use the provided training script with a configuration file. The current available configurations are:
```bash
# Train using nvblox and voxel-wise TSDF error
python scripts/train.py --config configs/nvblox_voxelwise_tsdf_error.yaml

# Train using nvblox and Chamfer distance
python scripts/train.py --config configs/nvblox_chamfer_distance.yaml

# Train using Open3D and Chamfer distance
python scripts/train.py --config configs/open3d_chamfer_distance.yaml
```

You can also pass multiple configuration files to use debug or demo mode (in case of conflicting parameters, the last config file would take precedence) and provide a checkpoint to resume training:
```bash
# Minimal run for debugging
python scripts/train.py --config configs/nvblox_voxelwise_tsdf_error.yaml configs/debug.yaml

# Minimal run with increased resolution for demo purposes
python scripts/train.py --config configs/open3d_chamfer_distance.yaml configs/debug.yaml configs/demo.yaml

# Resume training from a checkpoint
python scripts/train.py --config configs/nvblox_voxelwise_tsdf_error.yaml --checkpoint path/to/checkpoint.zip
```

### Create episode videos
To create videos of trained episodes, use the `create_episode_video.py` script:
```bash
python scripts/create_episode_videos.py path/to/eval_data/episode_xxxx
```
You can modify the script to change the camera views and frame rate.

## Known issues and fixes:
I used a CPU-only laptop and a Google Cloud VM with NVIDIA T4 GPU and `pytorch-2-7-cu128-ubuntu‑2404‑nvidia‑570` image. The following fixes were necessary on the VM.

**1. OpenCV/libGL error** (`libGL.so.1: cannot open shared object file`):
```bash
sudo apt install -y libgl1
```

**2. EGL error** (NVIDIA GPU offscreen rendering backend):
```bash
# Mesa EGL libraries (base requirement)
sudo apt install -y libegl1-mesa-dev libgles2-mesa-dev

# NVIDIA EGL (required for headless GPU rendering with NVIDIA drivers)
# Replace 570 with your driver version (check with: nvidia-smi)
sudo apt install -y libnvidia-gl-570-server
```

**3. MuJoCo rendering backend** (set EGL for headless/offscreen rendering):
```bash
export MUJOCO_GL=egl
# Add to ~/.bashrc for persistence:
echo 'export MUJOCO_GL=egl' >> ~/.bashrc
```

**4. EGL device enumeration fails with Mesa** (`Cannot initialize a EGL device display`):
If you have both NVIDIA and Mesa EGL drivers installed, the device enumeration may fail on Mesa devices. Force NVIDIA-only EGL:
```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
# Add to ~/.bashrc for persistence:
echo 'export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json' >> ~/.bashrc
```

**5. GPU render device permission denied** (`failed to open /dev/dri/renderD128: Permission denied`):
```bash
sudo usermod -aG render $USER
sudo usermod -aG video $USER
# Log out and back in, or use newgrp to apply immediately:
newgrp render
newgrp video
```

**6. Render images from robosuite are upside down** (e.g. when using EGL backend):
```bash
python external/robosuite/robosuite/scripts/setup_macros.py
```
In the created file external/robosuite/robosuite/macros_private.py switch from opengl to opencv convention:
```python
IMAGE_CONVENTION = "opencv"  # Options are {"opengl", "opencv"}
```

**Segmentation fault after several thousand steps** 

`pgrep -f "train.py"
[1]+  Segmentation fault      (core dumped) nohup python scripts/train.py --config configs/nvblox_voxelwise_tsdf_error.yaml > nvblox_voxelwise_tsdf_error.log 2>&1`

No fix found yet. Workaround: Restart training from last checkpoint.

TODO: faulthandler stack trace
    check RAM increase
hard coded policy