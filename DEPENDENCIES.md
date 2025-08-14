# Dependencies Guide

This document provides comprehensive information about all dependencies required for the JPEG compression experiment, including system-level dependencies for clean Ubuntu installations.

## 🖥️ System Requirements

### Operating System
- **Primary**: Ubuntu 20.04 LTS or later
- **Alternative**: Other Linux distributions (with package manager adjustments)
- **Windows/macOS**: Should work but not extensively tested

### Hardware Requirements
- **Memory**: Minimum 8GB RAM, recommended 16GB+
- **Storage**: ~10GB free disk space (models, results, datasets)
- **GPU**: NVIDIA GPU with CUDA support recommended (optional)
- **CPU**: Multi-core processor recommended for faster training/evaluation

### Python Version
- **Required**: Python 3.8 or later
- **Tested**: Python 3.8.10
- **Recommended**: Python 3.9 or 3.10 for best compatibility

## 📦 System-Level Dependencies (Ubuntu)

For a clean Ubuntu installation, install these packages first:

```bash
# Update package list
sudo apt update

# Essential build tools
sudo apt install -y \
    build-essential \
    git \
    wget \
    curl \
    unzip \
    software-properties-common

# Python development packages
sudo apt install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv

# Graphics and multimedia libraries
sudo apt install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# For OpenCV and image processing
sudo apt install -y \
    libopencv-dev \
    python3-opencv \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev

# For MuJoCo physics simulation
sudo apt install -y \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libglfw3 \
    libglfw3-dev

# For Atari environments
sudo apt install -y \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libasound2-dev \
    libpulse-dev \
    libaudio-dev \
    libx11-dev \
    libxext-dev \
    libxrandr-dev \
    libxcursor-dev \
    libxi-dev \
    libxinerama-dev \
    libxxf86vm-dev \
    libxss-dev \
    libgl1-mesa-dev \
    libdbus-1-dev \
    libudev-dev \
    libgles2-mesa-dev

# Additional utilities
sudo apt install -y \
    ffmpeg \
    swig \
    cmake \
    zlib1g-dev
```

## 🐍 Python Virtual Environment Setup

It's strongly recommended to use a virtual environment:

```bash
# Create virtual environment
python3 -m venv venv-jpeg-experiment

# Activate virtual environment
source venv-jpeg-experiment/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

## 📚 Python Package Dependencies

### Core Dependencies Matrix

| Package | Version | Purpose | Installation Notes |
|---------|---------|---------|-------------------|
| stable-baselines3[extra] | 2.1.0 | Deep RL algorithms | Includes extra dependencies |
| gymnasium[atari,mujoco] | 0.29.1 | RL environments | With Atari and MuJoCo support |
| dm-control | 1.0.14 | DeepMind control environments | Complex physics simulation |
| torch | 2.0.1 | Deep learning framework | CPU/CUDA versions available |
| torchvision | 0.15.2 | Computer vision utilities | Must match PyTorch version |
| Pillow | 10.0.0 | Image processing | JPEG compression/decompression |
| opencv-python | 4.8.1.78 | Computer vision | Image manipulation |
| numpy | 1.24.3 | Numerical computing | Array operations |
| pandas | 2.0.3 | Data analysis | Results processing |
| matplotlib | 3.7.2 | Plotting | Visualisation |
| seaborn | 0.12.2 | Statistical visualisation | Enhanced plots |
| tqdm | 4.65.0 | Progress bars | User feedback |
| huggingface-hub | 0.16.4 | Model downloading | Pre-trained model access |
| rl_zoo3 | latest | RL model zoo | Pre-trained agents |

### Installation Command
```bash
pip install -r requirements.txt
```

### Manual Installation (if requirements.txt fails)
```bash
# Core Deep RL
pip install stable-baselines3[extra]==2.1.0
pip install gymnasium[atari,mujoco]==0.29.1
pip install dm-control==1.0.14

# PyTorch (CPU version)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# PyTorch (CUDA 11.8 version - if you have NVIDIA GPU)
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Image processing
pip install Pillow==10.0.0 opencv-python==4.8.1.78

# Data analysis and visualisation
pip install numpy==1.24.3 pandas==2.0.3 matplotlib==3.7.2 seaborn==0.12.2

# Utilities
pip install tqdm==4.65.0 huggingface-hub==0.16.4

# RL Zoo
pip install rl_zoo3

# Additional environment dependencies
pip install ale-py==0.8.1
pip install "autorom[accept-rom-license]"==0.4.2
```

## 🎮 Environment-Specific Dependencies

### Atari Environments
```bash
# Install ALE and ROMs
pip install ale-py==0.8.1
pip install "autorom[accept-rom-license]"==0.4.2

# Download Atari ROMs (automatically accepts license)
python -c "import ale_py.roms as roms; roms.resolve()"
```

### MuJoCo Environments
```bash
# Install MuJoCo
pip install mujoco==2.3.7

# For gymnasium MuJoCo environments
pip install gymnasium[mujoco]

# Test installation
python -c "import gymnasium as gym; env = gym.make('Humanoid-v4'); print('MuJoCo working!')"
```

### DMControl Environments
```bash
# Install DMControl
pip install dm-control==1.0.14

# Test installation
python -c "from dm_control import suite; env = suite.load('cartpole', 'swingup'); print('DMControl working!')"
```

### CarRacing Environment
```bash
# Install Box2D for CarRacing
pip install gymnasium[box2d]
pip install Box2D==2.3.10

# Test installation
python -c "import gymnasium as gym; env = gym.make('CarRacing-v2'); print('CarRacing working!')"
```

## 🔧 GPU Support (Optional but Recommended)

### NVIDIA CUDA Setup
```bash
# Check if CUDA is available
nvidia-smi

# Install CUDA toolkit (if not already installed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Install CUDA-enabled PyTorch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

### Verify GPU Setup
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
```

## 🧪 Testing Installation

### Quick Test Script
```python
# Save as test_installation.py
import sys
import traceback

def test_import(module_name, package_name=None):
    try:
        if package_name:
            exec(f"from {module_name} import {package_name}")
        else:
            exec(f"import {module_name}")
        print(f"✓ {module_name} - OK")
        return True
    except Exception as e:
        print(f"✗ {module_name} - FAILED: {e}")
        return False

# Test core dependencies
tests = [
    ("numpy", None),
    ("pandas", None), 
    ("matplotlib.pyplot", None),
    ("PIL", "Image"),
    ("cv2", None),
    ("gymnasium", None),
    ("stable_baselines3", None),
    ("torch", None),
    ("rl_zoo3", None)
]

print("Testing installation...")
passed = 0
total = len(tests)

for module, package in tests:
    if test_import(module, package):
        passed += 1

print(f"\nResults: {passed}/{total} tests passed")

if passed == total:
    print("✅ All tests passed! Installation is ready.")
else:
    print("❌ Some tests failed. Check error messages above.")
```

Run the test:
```bash
python test_installation.py
```

## 🐋 Docker Alternative (Advanced)

For reproducible environments, consider using Docker:

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    libgomp1 git wget curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy experiment code
COPY . .

# Default command
CMD ["python3", "run_experiment.py"]
```

Build and run:
```bash
docker build -t jpeg-rl-experiment .
docker run --gpus all -v $(pwd):/workspace jpeg-rl-experiment
```

## 🔍 Troubleshooting Common Issues

### Issue: "No module named 'gymnasium'"
**Solution**:
```bash
pip install gymnasium[atari,mujoco]
```

### Issue: "MuJoCo not found"
**Solution**:
```bash
pip install mujoco==2.3.7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib/python3.8/site-packages/mujoco/bin
```

### Issue: "OpenGL context failed"
**Solution**:
```bash
sudo apt install libgl1-mesa-glx libglib2.0-0
export DISPLAY=:0
```

### Issue: "CUDA out of memory"
**Solution**:
- Reduce batch sizes in configuration
- Use CPU-only version: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

### Issue: "Permission denied" when downloading models
**Solution**:
```bash
pip install --upgrade huggingface-hub
huggingface-cli login  # If using private models
```

### Issue: "Atari ROM not found"
**Solution**:
```bash
pip install "autorom[accept-rom-license]"
python -c "import ale_py.roms as roms; roms.resolve()"
```

## 📋 Verification Checklist

Before running the experiment, ensure:

- [ ] Python 3.8+ installed and accessible
- [ ] Virtual environment created and activated  
- [ ] All system dependencies installed (Ubuntu packages)
- [ ] Python packages installed successfully
- [ ] GPU drivers and CUDA installed (if using GPU)
- [ ] Atari ROMs downloaded and accessible
- [ ] MuJoCo physics engine working
- [ ] DMControl environments loadable
- [ ] Test script passes all checks
- [ ] Sufficient disk space available (~10GB)
- [ ] Internet connection for model downloads

## 🆘 Getting Help

If you encounter issues:

1. **Check Prerequisites**: Ensure all system dependencies are installed
2. **Verify Python Version**: Must be 3.8 or later
3. **Update Package Managers**: `sudo apt update && pip install --upgrade pip`
4. **Use Virtual Environment**: Isolate dependencies from system packages
5. **Check GPU Setup**: Verify CUDA installation if using GPU
6. **Test Components**: Use the provided test scripts
7. **Check Logs**: Look in `logs/` directory for detailed error messages
8. **Environment Variables**: Ensure `LD_LIBRARY_PATH` includes MuJoCo
9. **Permissions**: Verify write permissions in experiment directory
10. **Internet Access**: Required for downloading models and packages

## 📝 Version Compatibility

### Known Working Combinations

**Ubuntu 20.04 LTS + Python 3.8.10**:
- All packages as specified in requirements.txt
- CUDA 11.8 + PyTorch 2.0.1
- Tested and verified working

**Ubuntu 22.04 LTS + Python 3.10**:
- Same package versions
- Minor adjustments may be needed for system packages

**Other Configurations**:
- Python 3.9: Should work with same package versions
- Python 3.11: May require newer package versions
- Windows: Not extensively tested, may need conda instead of pip

### Package Version Constraints

- **PyTorch**: Must match CUDA version if using GPU
- **Gymnasium**: Newer versions may have breaking changes
- **Stable-Baselines3**: Specific version for model compatibility
- **NumPy**: Version constraints from other packages
- **Pillow**: Must support JPEG compression features

---

**Last Updated**: August 2025  
**Tested On**: Ubuntu 20.04 LTS with Python 3.8.10  
**Total Installation Time**: ~30-60 minutes (depending on internet speed)