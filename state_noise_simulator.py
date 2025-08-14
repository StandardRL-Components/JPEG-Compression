#!/usr/bin/env python3
"""
State noise simulator to approximate the effect of visual compression on state observations.
Since we can't directly apply JPEG compression to state-based models, we simulate the effect
by adding noise to state observations that corresponds to visual compression levels.
"""

import numpy as np
from compression_utils import JPEGCompressor

class StateNoiseSimulator:
    """Simulates the effect of JPEG compression by adding noise to state observations."""
    
    def __init__(self, compression_level='none'):
        self.compression_level = compression_level
        
        # Map compression levels to noise levels
        # These values simulate information loss from visual compression
        self.noise_mapping = {
            'none': 0.0,        # No noise - baseline
            'minimal': 0.001,   # Very small noise - 95% JPEG quality
            'low': 0.005,       # Small noise - 75% JPEG quality  
            'medium': 0.02,     # Medium noise - 50% JPEG quality
            'high': 0.05,       # High noise - 30% JPEG quality
            'very_high': 0.1    # Very high noise - 10% JPEG quality
        }
        
        self.noise_std = self.noise_mapping.get(compression_level, 0.0)
        
    def add_compression_noise(self, state_obs):
        """Add noise to state observation to simulate compression effects."""
        if self.noise_std == 0.0:
            return state_obs
        
        # Add Gaussian noise scaled by observation magnitude
        noise = np.random.normal(0, self.noise_std, state_obs.shape)
        
        # Scale noise by the observation values to make it relative
        scaled_noise = noise * (np.abs(state_obs) + 0.1)  # +0.1 to avoid zero scaling
        
        return state_obs + scaled_noise.astype(state_obs.dtype)
    
    def get_compression_ratio(self):
        """Get approximate compression ratio for this noise level."""
        # Approximate ratios based on typical JPEG compression
        ratio_mapping = {
            'none': 1.0,
            'minimal': 1.2,
            'low': 2.0,
            'medium': 3.0,
            'high': 5.0,
            'very_high': 10.0
        }
        return ratio_mapping.get(self.compression_level, 1.0)
    
    def __str__(self):
        return f"StateNoiseSimulator(level={self.compression_level}, std={self.noise_std})"