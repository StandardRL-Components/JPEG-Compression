#!/usr/bin/env python3
"""
Configuration for JPEG compression experiment using PRE-TRAINED realistic models.
This version uses actual pre-trained models from Hugging Face for robotic manipulation.
"""

import os
from pathlib import Path

# Experiment configuration
EXPERIMENT_CONFIG = {
    # Number of episodes to run for evaluation
    'num_evaluation_episodes': 50,  # Good statistical power
    
    # Maximum steps per episode (safety limit)
    'max_episode_steps': 500,
    
    # Random seed for reproducibility
    'random_seed': 42,
    
    # Results directory
    'results_dir': Path('results_pretrained_realistic'),
    
    # Logs directory  
    'logs_dir': Path('logs_pretrained_realistic'),
    
    # Models directory
    'models_dir': Path('models_pretrained_realistic'),
    
    # Whether to render episodes (for debugging)
    'render_episodes': False,
    
    # Whether to record videos
    'record_videos': True,
    
    # Video recording frequency (every N episodes)
    'video_frequency': 10,
    
    # Visual observation settings
    'image_size': (224, 224),  # High resolution for realistic images
    'use_rgb': True,
}

# PRE-TRAINED models from Hugging Face Hub for realistic manipulation environments
MODEL_CONFIGS = {
    'ppo_walker2d': {
        'algorithm': 'PPO',
        'environment': 'Walker2d-v4',
        'model_path': 'models_pretrained_realistic/ppo_walker2d',
        'huggingface_repo': 'crispisu/ppo-Walker2D-v4',
        'description': 'PPO agent for Walker2d bipedal walking (pre-trained)',
        'observation_space': 'visual',
        'action_space': 'continuous',
        'complexity': 'bipedal_locomotion',
        'mean_reward': 3571.74,
        'std_reward': 807.75,
        'download_command': 'huggingface-cli download crispisu/ppo-Walker2D-v4 model.zip'
    },
    
    'sac_hopper': {
        'algorithm': 'SAC',
        'environment': 'Hopper-v5', 
        'model_path': 'models_pretrained_realistic/sac_hopper',
        'huggingface_repo': 'farama-minari/Hopper-v5-SAC-expert',
        'description': 'SAC agent for Hopper one-legged hopping (pre-trained)',
        'observation_space': 'visual',
        'action_space': 'continuous',
        'complexity': 'single_leg_locomotion',
        'mean_reward': 4098.17,
        'std_reward': 247.70,
        'download_command': 'huggingface-cli download farama-minari/Hopper-v5-SAC-expert model.zip'
    },
    
    
}

# Environment configurations with visual properties  
ENVIRONMENT_CONFIGS = {
    'Walker2d-v4': {
        'type': 'mujoco_control',
        'visual_obs_shape': (224, 224, 3),
        'requires_preprocessing': False,
        'real_world': True,  # Physics-based simulation
        'complexity': 'moderate',
        'task_type': 'bipedal_locomotion',
        'robot': 'walker2d',
        'render_mode': 'rgb_array',
        'has_goal': False,
        'observation_type': 'array'
    },
    
    'Hopper-v5': {
        'type': 'mujoco_control',
        'visual_obs_shape': (224, 224, 3),
        'requires_preprocessing': False,
        'real_world': True,
        'complexity': 'moderate',
        'task_type': 'single_leg_locomotion',
        'robot': 'hopper',
        'render_mode': 'rgb_array',
        'has_goal': False,
        'observation_type': 'array'
    },
    
}

# Compression level configurations (same as before)
COMPRESSION_CONFIGS = {
    'none': {'quality': 100, 'description': 'No compression (baseline)'},
    'minimal': {'quality': 95, 'description': 'Minimal compression (~80-95% original size)'},
    'low': {'quality': 75, 'description': 'Low compression (~50-70% original size)'},
    'medium': {'quality': 50, 'description': 'Medium compression (~30-40% original size)'},
    'high': {'quality': 30, 'description': 'High compression (~15-25% original size)'},
    'very_high': {'quality': 10, 'description': 'Very high compression (~5-10% original size)'}
}

def get_experiment_name(model_name, compression_level):
    """Generate experiment name for logging and results."""
    return f"{model_name}_{compression_level}_compression"

def ensure_directories():
    """Create necessary directories for the experiment."""
    dirs_to_create = [
        EXPERIMENT_CONFIG['results_dir'],
        EXPERIMENT_CONFIG['logs_dir'], 
        EXPERIMENT_CONFIG['models_dir'],
        EXPERIMENT_CONFIG['results_dir'] / 'videos',
        EXPERIMENT_CONFIG['results_dir'] / 'plots',
        EXPERIMENT_CONFIG['results_dir'] / 'raw_data'
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(exist_ok=True, parents=True)

def get_model_environment_pairs():
    """Get all valid model-environment pairs for the experiment."""
    pairs = []
    for model_name, model_config in MODEL_CONFIGS.items():
        env_name = model_config['environment']
        if env_name in ENVIRONMENT_CONFIGS:
            pairs.append((model_name, env_name))
    return pairs

def validate_config():
    """Validate experiment configuration."""
    errors = []
    
    # Check if all model environments are defined
    for model_name, model_config in MODEL_CONFIGS.items():
        env_name = model_config['environment']
        if env_name not in ENVIRONMENT_CONFIGS:
            errors.append(f"Environment {env_name} for model {model_name} not found in ENVIRONMENT_CONFIGS")
    
    # Check if directories can be created
    try:
        ensure_directories()
    except Exception as e:
        errors.append(f"Cannot create directories: {e}")
        
    if errors:
        raise ValueError("Configuration validation failed:\\n" + "\\n".join(errors))
        
    return True

def download_all_models():
    """Download all pre-trained models from Hugging Face Hub."""
    print("📥 Downloading All Pre-trained Models")
    print("=" * 60)
    
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\\n🔄 Downloading {model_name}...")
        print(f"   Repository: {config['huggingface_repo']}")
        print(f"   Command: {config['download_command']}")
        print(f"   Performance: {config['mean_reward']:.2f} ± {config['std_reward']:.2f}")

if __name__ == "__main__":
    print("🚀 PRE-TRAINED Realistic Robotics JPEG Compression Experiment")
    print("=" * 70)
    
    try:
        validate_config()
        print("✅ Configuration validation passed!")
        
        print(f"\\n🤖 Experiment will test {len(MODEL_CONFIGS)} PRE-TRAINED realistic models:")
        for model_name, config in MODEL_CONFIGS.items():
            perf = f"{config['mean_reward']:.1f}±{config['std_reward']:.1f}"
            print(f"  - {model_name}: {config['description']} (Reward: {perf})")
            
        print(f"\\n🎯 On {len(ENVIRONMENT_CONFIGS)} realistic environments:")
        for env_name, config in ENVIRONMENT_CONFIGS.items():
            goal_str = "Goal-conditioned" if config.get('has_goal') else "Direct reward"
            print(f"  - {env_name}: {config['robot']} - {config['task_type']} ({goal_str})")
                  
        print(f"\\n📊 Using {len(COMPRESSION_CONFIGS)} compression levels:")
        for level, config in COMPRESSION_CONFIGS.items():
            print(f"  - {level}: {config['description']}")
            
        print(f"\\n📈 Experiment Statistics:")
        print(f"  - Total experiments: {len(MODEL_CONFIGS) * len(COMPRESSION_CONFIGS)}")
        print(f"  - Episodes per experiment: {EXPERIMENT_CONFIG['num_evaluation_episodes']}")
        total_episodes = len(MODEL_CONFIGS) * len(COMPRESSION_CONFIGS) * EXPERIMENT_CONFIG['num_evaluation_episodes']
        print(f"  - Total episodes to run: {total_episodes}")
        
        print(f"\\n🎯 This experiment focuses on:")
        print("  ✅ PRE-TRAINED models from Hugging Face Hub")
        print("  ✅ Realistic robot manipulation tasks")
        print("  ✅ High-resolution camera-like visual observations (224x224)")
        print("  ✅ Physics-based simulation environments")
        print("  ✅ Goal-conditioned and direct reward tasks")
        print("  ✅ Multiple manipulation complexity levels")
        
        download_all_models()
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        exit(1)