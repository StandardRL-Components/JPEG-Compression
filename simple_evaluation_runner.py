#!/usr/bin/env python3
"""
Simplified evaluation runner for the realistic JPEG compression experiment.
"""

import json
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

class EvaluationRunner:
    """Simplified evaluation runner for realistic experiments."""
    
    def __init__(self, config):
        """Initialize with experiment configuration."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def evaluate_model_with_compression(self, model, env, compressor, experiment_name):
        """
        Evaluate a model with simulated compression noise applied to state observations.
        
        Since we're using pre-trained state-based models, we simulate the effect of 
        visual compression by adding corresponding noise to state observations.
        
        Args:
            model: Pre-trained RL model
            env: Original state-based environment  
            compressor: JPEG compressor (used for compression level)
            experiment_name: Name for this experiment
            
        Returns:
            dict: Evaluation results
        """
        from state_noise_simulator import StateNoiseSimulator
        
        num_episodes = self.config.get('num_evaluation_episodes', 50)
        episode_rewards = []
        episode_lengths = []
        
        self.logger.info(f"Starting evaluation: {experiment_name}")
        
        # Create noise simulator based on compression level
        compression_level = experiment_name.split('_')[-2]  # Extract compression level from name
        noise_simulator = StateNoiseSimulator(compression_level)
        compression_ratio = noise_simulator.get_compression_ratio()
        
        self.logger.info(f"Using noise level: {noise_simulator.noise_std:.4f} (simulating {compression_level} compression)")
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0
            episode_length = 0
            
            done = False
            truncated = False
            
            while not (done or truncated):
                # Apply compression-equivalent noise to state observation
                noisy_obs = noise_simulator.add_compression_noise(obs)
                
                # Get action from model using noisy observation (simulates compression effect)
                action, _ = model.predict(noisy_obs, deterministic=True)
                
                # Step environment with clean observation (environment uses clean state)
                obs, reward, done, truncated, info = env.step(action)
                
                total_reward += reward
                episode_length += 1
                
                # Safety limit
                if episode_length >= self.config.get('max_episode_steps', 500):
                    truncated = True
            
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            
            if (episode + 1) % 10 == 0:
                mean_reward = np.mean(episode_rewards)
                self.logger.info(f"  Episode {episode + 1}/{num_episodes}: Mean reward = {mean_reward:.2f}")
        
        # Calculate final statistics
        results = {
            'experiment_name': experiment_name,
            'model_name': experiment_name.split('_')[0] + '_' + experiment_name.split('_')[1],
            'compression_level': compression_level,
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'min_reward': float(np.min(episode_rewards)),
            'max_reward': float(np.max(episode_rewards)),
            'mean_episode_length': float(np.mean(episode_lengths)),
            'compression_ratio': float(compression_ratio),
            'noise_std': float(noise_simulator.noise_std),
            'num_episodes_completed': len(episode_rewards),
            'completion_rate': len(episode_rewards) / num_episodes,
            'episode_rewards': [float(r) for r in episode_rewards],
            'episode_lengths': [int(l) for l in episode_lengths],
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Completed {experiment_name}: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        return results
    
    def save_intermediate_results(self, all_results):
        """Save intermediate results during experiment."""
        results_dir = Path(self.config.get('results_dir', 'results_pretrained_realistic'))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = results_dir / 'raw_data' / f'intermediate_results_{timestamp}.json'
        filename.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    def save_experiment_results(self, all_results, results_dir):
        """Save final experiment results."""
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(results_dir / 'experiment_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create summary
        summary = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'total_experiments': len(all_results),
                'successful_experiments': len([r for r in all_results.values() if r.get('completion_rate', 0) > 0])
            },
            'results_summary': {}
        }
        
        for exp_name, results in all_results.items():
            summary['results_summary'][exp_name] = {
                'mean_reward': results['mean_reward'],
                'std_reward': results['std_reward'],
                'compression_ratio': results['compression_ratio'],
                'completion_rate': results['completion_rate']
            }
        
        with open(results_dir / 'experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Results saved to {results_dir}")

if __name__ == "__main__":
    # Simple test
    config = {'num_evaluation_episodes': 3}
    runner = EvaluationRunner(config)
    print("✅ Simple evaluation runner created successfully")