# JPEG Compression Robustness Experiment - Complete Instructions

## Overview

This experiment tests how JPEG compression-equivalent noise affects pre-trained reinforcement learning models in realistic robotic environments. It uses state noise simulation to approximate the effects of visual compression on model performance.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run experiment (downloads models automatically)
python run_realistic_experiment.py

# 3. View results
ls results_pretrained_realistic/
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- ~1GB free disk space for models and results

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `gymnasium[mujoco]>=1.0.0` - RL environment framework
- `mujoco>=3.0.0` - Physics simulation
- `stable-baselines3>=2.0.0` - RL algorithms
- `sb3-contrib` - Additional algorithms (TQC, etc.)
- `huggingface-hub>=0.17.0` - Model downloads
- `matplotlib>=3.5.0` - Plotting
- `pandas>=1.5.0` - Data analysis
- `pillow>=9.0.0` - Image processing
- `numpy>=1.21.0` - Numerical computation

### Step 2: Verify Installation

```bash
python -c "
import gymnasium as gym
import mujoco
import stable_baselines3
print('✅ All packages installed successfully')
"
```

## 🤖 Models Used

The experiment automatically downloads these pre-trained models from HuggingFace Hub:

### 1. PPO Walker2d-v4
- **Repository**: `crispisu/ppo-Walker2D-v4`
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Environment**: Walker2d-v4 (bipedal walking)
- **File**: `ppo-Walker2d-v4.zip` (~163 KB)
- **Expected Performance**: ~1100 reward

### 2. SAC Hopper-v5
- **Repository**: `farama-minari/Hopper-v5-SAC-expert`
- **Algorithm**: Soft Actor-Critic (SAC)
- **Environment**: Hopper-v5 (one-legged hopping)
- **File**: `hopper-v5-SAC-expert.zip` (~3 MB)
- **Expected Performance**: ~1960 reward

**Note**: Models are downloaded automatically on first run to `models_pretrained_realistic/`

## 🚀 Running the Experiment

### Full Experiment

```bash
python run_realistic_experiment.py
```

**What happens:**
- Downloads models (first run only)
- Tests 6 compression levels per model
- Runs 50 episodes per configuration
- Total: 12 experiments, 600 episodes
- Runtime: ~5-10 minutes

### Experiment Configuration

The experiment tests these compression levels:

| Level | Noise STD | Compression Ratio | Description |
|-------|-----------|-------------------|-------------|
| none | 0.000 | 1.0x | Baseline (no noise) |
| minimal | 0.001 | 1.2x | Minimal compression |
| low | 0.005 | 2.0x | Low compression |
| medium | 0.020 | 3.0x | Medium compression |
| high | 0.050 | 5.0x | High compression |
| very_high | 0.100 | 10.0x | Very high compression |

## 📊 Interpreting Results

### Output Files

After completion, results are saved to `results_pretrained_realistic/`:

```
results_pretrained_realistic/
├── experiment_results.json          # Raw experiment data
├── experiment_summary.json          # Summary statistics  
├── experiment_results.csv           # Data in CSV format
├── summary_statistics.csv           # Performance analysis
└── plots/
    ├── reward_by_compression.png           # Bar chart: reward vs compression
    ├── episode_length_by_compression.png   # Bar chart: episode length vs compression
    └── performance_vs_compression_ratio.png # Performance drop analysis
```

### Key Metrics

**experiment_results.csv** contains:
- `model`: Model name (ppo_walker2d, sac_hopper)
- `compression_level`: Compression level tested
- `mean_reward`: Average reward over 50 episodes
- `std_reward`: Standard deviation of rewards
- `mean_episode_length`: Average episode length
- `compression_ratio`: Simulated compression ratio
- `noise_std`: Noise standard deviation applied

**summary_statistics.csv** contains:
- `performance_drop_percent`: Performance drop from baseline
- `baseline_reward`: Baseline performance (no compression)

### Understanding the Results

**Performance Patterns:**

1. **Robust Model (PPO Walker2d)**:
   - Minimal performance change across compression levels
   - May show slight improvement due to regularisation effect
   - Indicates good noise tolerance

2. **Sensitive Model (SAC Hopper)**:
   - Clear performance degradation with higher compression
   - Sharp drop at medium-high compression levels
   - High variance in performance at high compression

### Example Interpretation

```csv
model,compression_level,mean_reward,performance_drop_percent
sac_hopper,none,1964,0.0
sac_hopper,medium,1830,-6.8
sac_hopper,high,895,-54.4
```

**Interpretation**: SAC Hopper maintains performance until medium compression but shows catastrophic failure at high compression (54% performance drop).

## 🔬 Scientific Analysis

### Hypothesis Testing

**Null Hypothesis (H₀)**: Compression noise has no effect on RL model performance  
**Alternative Hypothesis (H₁)**: Compression noise causes performance degradation

### Statistical Significance

The experiment runs 50 episodes per configuration for statistical power. Key indicators:

- **Standard deviation**: Higher values indicate less stable performance
- **Performance drop**: Percentage change from baseline
- **Compression threshold**: Level where significant degradation begins

### Expected Results

**Robust Algorithm Pattern**:
- Flat performance across compression levels
- Low standard deviation
- No significant compression threshold

**Sensitive Algorithm Pattern**:
- Clear negative trend with compression
- Increasing standard deviation at high compression
- Identifiable compression threshold (e.g., 50% JPEG quality)

## 🛠️ Troubleshooting

### Common Issues

**1. ImportError: No module named 'mujoco'**
```bash
pip install mujoco>=3.0.0
```

**2. Environment creation fails**
```bash
# Check MuJoCo installation
python -c "import mujoco; print('✅ MuJoCo OK')"
```

**3. Model download fails**
```bash
# Check internet connection and try manual download
pip install huggingface-hub
huggingface-cli download farama-minari/Hopper-v5-SAC-expert model.zip
```

**4. CUDA/GPU issues**
```bash
# The experiment works on CPU, GPU not required
# For faster training/evaluation, ensure CUDA is properly installed
```

**5. Permission denied errors**
```bash
# Ensure write permissions for results directory
mkdir -p results_pretrained_realistic
```

### System Requirements

**Minimum**:
- 4GB RAM
- 1GB free disk space
- Python 3.8+

**Recommended**:
- 8GB RAM
- CUDA GPU
- SSD storage

## 📈 Advanced Usage

### Custom Configuration

Edit `pretrained_realistic_config.py` to:
- Change number of episodes per experiment
- Modify compression levels
- Add new models
- Adjust noise simulation parameters

### Adding New Models

1. Find a compatible model on HuggingFace Hub
2. Add to `MODEL_CONFIGS` in `pretrained_realistic_config.py`
3. Update filename mapping in `pretrained_model_loader.py`

### Custom Analysis

Use the generated CSV files for custom analysis:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results_pretrained_realistic/experiment_results.csv')

# Custom analysis
for model in df['model'].unique():
    model_data = df[df['model'] == model]
    plt.plot(model_data['noise_std'], model_data['mean_reward'], 
             label=model, marker='o')

plt.xlabel('Noise Standard Deviation')
plt.ylabel('Mean Reward')
plt.legend()
plt.show()
```

## 🎯 Research Applications

This experiment framework can be used for:

1. **Algorithm robustness testing**: Compare RL algorithms' noise tolerance
2. **Deployment planning**: Determine acceptable compression levels for robotics
3. **Architecture research**: Study which model architectures are more robust
4. **Compression optimisation**: Find optimal compression-performance trade-offs

## ⚠️ Limitations

1. **State noise simulation**: Approximates but doesn't perfectly replicate visual compression effects
2. **Limited environments**: Only tests on locomotion tasks
3. **Fixed models**: Uses existing pre-trained models, doesn't test during training
4. **Gaussian noise**: Real compression artifacts may have different statistical properties

## 📚 References

- **Stable Baselines3**: https://stable-baselines3.readthedocs.io/
- **Gymnasium**: https://gymnasium.farama.org/
- **MuJoCo**: https://mujoco.readthedocs.io/
- **HuggingFace Hub**: https://huggingface.co/docs/hub/

---

**For questions or issues, check the troubleshooting section or examine the generated log files in the results directory.**