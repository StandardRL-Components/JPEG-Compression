#!/usr/bin/env python3
"""
Model loading utilities for PRE-TRAINED realistic visual RL models.
Downloads and loads existing trained models from Hugging Face Hub.
"""

import os
import sys
import numpy as np
from pathlib import Path
import logging
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import RecordVideo
import warnings
import subprocess
warnings.filterwarnings('ignore')

# Set MuJoCo rendering backend
os.environ['MUJOCO_GL'] = 'egl'

# Stable Baselines3 imports
try:
    from stable_baselines3 import DQN, PPO, SAC, A2C
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    from stable_baselines3.common.evaluation import evaluate_policy
    from sb3_contrib import TQC  # For TQC models
except ImportError as e:
    print(f"Error importing stable-baselines3: {e}")
    print("Please install with: pip install stable-baselines3[extra] sb3-contrib")
    sys.exit(1)

# RL Zoo imports for model loading
try:
    from rl_zoo3 import load_from_hub
except ImportError as e:
    print(f"Error importing rl_zoo3: {e}")
    print("Please install with: pip install rl_zoo3")
    sys.exit(1)

# Note: PyBullet removed due to compatibility issues with Gymnasium 1.0.0

from pretrained_realistic_config import MODEL_CONFIGS, ENVIRONMENT_CONFIGS

class VisualWrapper(gym.ObservationWrapper):
    """Wrapper to convert state-based observations to visual for JPEG compression testing"""
    
    def __init__(self, env, image_size=(224, 224)):
        super().__init__(env)
        self.image_size = image_size
        
        # Store original observation space for reference
        self.original_observation_space = env.observation_space
        
        # Update observation space to visual only
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(*image_size, 3), 
            dtype=np.uint8
        )
    
    def observation(self, obs):
        # Get visual observation from rendering
        try:
            rgb_array = self.env.render()
            if rgb_array is not None:
                # Resize if needed
                if rgb_array.shape[:2] != self.image_size:
                    from PIL import Image
                    img = Image.fromarray(rgb_array.astype(np.uint8))
                    img = img.resize(self.image_size)
                    rgb_array = np.array(img)
                
                return rgb_array.astype(np.uint8)
            else:
                # Fallback: create black image
                return np.zeros((*self.image_size, 3), dtype=np.uint8)
                
        except Exception as e:
            logging.warning(f"Visual observation failed: {e}")
            # Fallback to black image
            return np.zeros((*self.image_size, 3), dtype=np.uint8)

class PretrainedModelLoader:
    """Handles downloading and loading of pre-trained realistic visual RL models."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def make_environment(self, env_name, render_mode='rgb_array'):
        """Create and configure realistic environment with visual observations."""
        self.logger.info(f"Creating realistic environment: {env_name}")
        
        env_config = ENVIRONMENT_CONFIGS.get(env_name, {})
        
        try:
            # Create base environment
            env = gym.make(env_name, render_mode=render_mode)
            
            # Wrap with visual observations for JPEG compression testing
            env = VisualWrapper(env, image_size=(224, 224))
            
            self.logger.info(f"Environment created successfully: {env_name}")
            self.logger.info(f"Visual observation space: {env.observation_space}")
            self.logger.info(f"Action space: {env.action_space}")
            
            return env
            
        except Exception as e:
            self.logger.error(f"Failed to create environment {env_name}: {e}")
            raise
            
    def download_model_from_hub(self, model_name):
        """Download pre-trained model from Hugging Face Hub."""
        model_config = MODEL_CONFIGS[model_name]
        
        try:
            self.logger.info(f"Downloading model from hub: {model_config['huggingface_repo']}")
            
            # Create model save directory
            model_path = Path(model_config['model_path'])
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Use huggingface_hub to download models
            from huggingface_hub import hf_hub_download
            
            repo_id = model_config['huggingface_repo']
            
            # Use specific filenames for each repository
            if repo_id == "farama-minari/Hopper-v5-SAC-expert":
                filename = "hopper-v5-SAC-expert.zip"
            elif repo_id == "crispisu/ppo-Walker2D-v4":
                filename = "ppo-Walker2d-v4.zip"
            else:
                # Fallback to generic naming
                filename = f"{model_config['algorithm'].lower()}-{model_config['environment']}.zip"
            
            self.logger.info(f"Downloading {filename} from {repo_id}")
            
            model_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(model_path),
                local_dir_use_symlinks=False  # Use actual files, not symlinks
            )
            model_file = Path(model_file)
            
            self.logger.info(f"Successfully downloaded model to: {model_file}")
            return model_file
            
        except Exception as e:
            self.logger.error(f"Failed to download model from hub: {e}")
            raise
            
    def load_model(self, model_name):
        """Load a pre-trained model from Hugging Face Hub."""
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_config = MODEL_CONFIGS[model_name]
        
        # Check if model is already downloaded
        model_path = Path(model_config['model_path'])
        model_files = list(model_path.rglob("*.zip"))
        
        if not model_files:
            # Download the model
            model_file = self.download_model_from_hub(model_name)
        else:
            # Use existing model
            model_file = model_files[0]  # Take first found model
            self.logger.info(f"Using existing model: {model_file}")
        
        # Create environment for model loading - IMPORTANT: use original state-based environment
        original_env = gym.make(model_config['environment'])
        
        # Load model based on algorithm
        algo = model_config['algorithm'].lower()
        
        try:
            if algo == 'tqc':
                model = TQC.load(str(model_file), env=original_env)
            elif algo == 'ppo':
                model = PPO.load(str(model_file), env=original_env)
            elif algo == 'sac':
                model = SAC.load(str(model_file), env=original_env)
            elif algo == 'dqn':
                model = DQN.load(str(model_file), env=original_env)
            else:
                raise ValueError(f"Unsupported algorithm: {algo}")
            
            self.logger.info(f"Successfully loaded pre-trained model: {model_name}")
            
            # Return both the model and the original environment
            # The visual wrapper will be applied during evaluation
            return model, original_env
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            original_env.close()
            raise

def test_pretrained_model_loading():
    """Test pre-trained model loading functionality."""
    print("🚀 Testing Pre-trained Model Loading")
    print("=" * 60)
    
    # Install required packages first
    print("📦 Installing required packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "sb3-contrib", "rl-zoo3", "pybullet"], 
                      check=True, capture_output=True)
        print("✅ Packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Package installation warning: {e}")
    
    loader = PretrainedModelLoader()
    
    # Test with one pre-trained model (standard MuJoCo model)
    test_model = 'ppo_pusher'
    try:
        print(f"\\n🔄 Testing {test_model}...")
        model, env = loader.load_model(test_model)
        
        print(f"✅ Successfully loaded {test_model}")
        print(f"  Model: {type(model)}")
        print(f"  Environment: {env.spec.id if hasattr(env, 'spec') else 'VisualWrapper'}")
        print(f"  Visual observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # Quick evaluation test
        obs, _ = env.reset()
        print(f"  Reset obs shape: {obs.shape}")
        
        action, _ = model.predict(obs)
        print(f"  Predicted action shape: {action.shape}")
        
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Test step successful - reward: {reward}")
        
        # Test JPEG compression simulation
        from PIL import Image
        img = Image.fromarray(obs.astype(np.uint8))
        img.save(f"test_{test_model}_observation.png")
        print(f"  Sample observation saved: test_{test_model}_observation.png")
        
        # Test compression
        compressed_img = img.copy()
        compressed_img.save(f"test_{test_model}_compressed.jpg", "JPEG", quality=30)
        print(f"  Compressed observation saved: test_{test_model}_compressed.jpg")
        
        env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load {test_model}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = test_pretrained_model_loading()
    exit(0 if success else 1)