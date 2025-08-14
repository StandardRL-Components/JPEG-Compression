#!/usr/bin/env python3
"""
JPEG compression utilities for Deep RL visual input experiment.
Handles compression/decompression pipeline with configurable quality levels.
"""

import io
import numpy as np
from PIL import Image
import cv2

class JPEGCompressor:
    """Handles JPEG compression and decompression of visual observations."""
    
    # Define 5 compression levels from very high compression to minimal compression
    COMPRESSION_LEVELS = {
        'very_high': {'quality': 10, 'description': 'Very high compression (~5-10% original size)'},
        'high': {'quality': 30, 'description': 'High compression (~15-25% original size)'},
        'medium': {'quality': 50, 'description': 'Medium compression (~30-40% original size)'},
        'low': {'quality': 75, 'description': 'Low compression (~50-70% original size)'},
        'minimal': {'quality': 95, 'description': 'Minimal compression (~80-95% original size)'},
        'none': {'quality': 100, 'description': 'No compression (original)'}
    }
    
    def __init__(self, quality_level='medium'):
        """
        Initialize compressor with specified quality level.
        
        Args:
            quality_level (str): One of 'very_high', 'high', 'medium', 'low', 'minimal', 'none'
        """
        if quality_level not in self.COMPRESSION_LEVELS:
            raise ValueError(f"Invalid quality level. Choose from: {list(self.COMPRESSION_LEVELS.keys())}")
        
        self.quality_level = quality_level
        self.quality = self.COMPRESSION_LEVELS[quality_level]['quality']
        self.description = self.COMPRESSION_LEVELS[quality_level]['description']
        
    def compress_observation(self, observation):
        """
        Compress a visual observation using JPEG compression.
        
        Args:
            observation (np.array): Visual observation, typically shape (H, W, C) or (C, H, W)
            
        Returns:
            np.array: Compressed and decompressed observation, same shape as input
        """
        if self.quality == 100:  # No compression
            return observation.copy()
            
        # Handle different input formats
        original_shape = observation.shape
        original_dtype = observation.dtype
        
        # Convert to uint8 if needed (required for JPEG)
        if observation.dtype != np.uint8:
            # Normalize to 0-255 range
            if observation.max() <= 1.0:
                obs_uint8 = (observation * 255).astype(np.uint8)
            else:
                obs_uint8 = observation.astype(np.uint8)
        else:
            obs_uint8 = observation
            
        # Handle different channel arrangements
        if len(original_shape) == 3:
            if original_shape[0] in [1, 3, 4]:  # CHW format (including 4 channels)
                obs_uint8 = np.transpose(obs_uint8, (1, 2, 0))
                was_chw = True
            else:  # HWC format
                was_chw = False
        elif len(original_shape) == 2:  # Grayscale
            was_chw = False
        else:
            raise ValueError(f"Unsupported observation shape: {original_shape}")
            
        # Handle 4-channel images (like stacked Atari frames) by converting to RGB
        if len(obs_uint8.shape) == 3 and obs_uint8.shape[2] == 4:
            # For 4-channel images, take the last 3 channels or convert to grayscale
            # Option 1: Take last 3 channels
            obs_uint8 = obs_uint8[:, :, 1:4]  # Skip first channel, take next 3
            
        # Convert to PIL Image
        if len(obs_uint8.shape) == 3 and obs_uint8.shape[2] == 3:
            # RGB image
            pil_image = Image.fromarray(obs_uint8, 'RGB')
        elif len(obs_uint8.shape) == 3 and obs_uint8.shape[2] == 1:
            # Grayscale with channel dimension
            pil_image = Image.fromarray(obs_uint8[:, :, 0], 'L')
        elif len(obs_uint8.shape) == 2:
            # Grayscale
            pil_image = Image.fromarray(obs_uint8, 'L')
        else:
            raise ValueError(f"Cannot convert observation shape {obs_uint8.shape} to PIL Image")
            
        # Compress using JPEG
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=self.quality, optimize=True)
        buffer.seek(0)
        
        # Decompress
        compressed_image = Image.open(buffer)
        
        # Convert back to numpy array
        compressed_array = np.array(compressed_image)
        
        # Restore original shape and channel arrangement
        if len(original_shape) == 3:
            if len(compressed_array.shape) == 2:  # Grayscale result
                if original_shape[2] == 1 or original_shape[0] == 1:
                    compressed_array = np.expand_dims(compressed_array, axis=-1)
                else:
                    # Convert grayscale to RGB
                    compressed_array = np.stack([compressed_array] * 3, axis=-1)
            
            # Handle 4-channel restoration
            if original_shape[2] == 4 or (was_chw and original_shape[0] == 4):
                # We compressed 4-channel to 3-channel, now restore 4th channel
                if len(compressed_array.shape) == 3 and compressed_array.shape[2] == 3:
                    # Add the first channel back (duplicate the first compressed channel)
                    first_channel = compressed_array[:, :, 0:1]  # Keep dims
                    compressed_array = np.concatenate([first_channel, compressed_array], axis=2)
            
            if was_chw and len(compressed_array.shape) == 3:
                compressed_array = np.transpose(compressed_array, (2, 0, 1))
                
        # Restore original data type
        if original_dtype != np.uint8:
            if observation.max() <= 1.0:
                compressed_array = compressed_array.astype(np.float32) / 255.0
            else:
                compressed_array = compressed_array.astype(original_dtype)
        
        # Ensure output shape matches input shape
        if compressed_array.shape != original_shape:
            # Handle potential shape mismatches
            if len(original_shape) == 3 and len(compressed_array.shape) == 2:
                compressed_array = np.expand_dims(compressed_array, axis=0 if was_chw else -1)
            elif original_shape != compressed_array.shape:
                print(f"Warning: Shape mismatch. Original: {original_shape}, Compressed: {compressed_array.shape}")
                # Try to reshape to match or pad/crop as needed
                try:
                    if len(original_shape) == 3 and len(compressed_array.shape) == 3:
                        # Handle channel dimension mismatch
                        h, w = original_shape[:2] if not was_chw else original_shape[1:3]
                        target_channels = original_shape[2] if not was_chw else original_shape[0]
                        current_channels = compressed_array.shape[2] if not was_chw else compressed_array.shape[0]
                        
                        if target_channels > current_channels:
                            # Pad channels by repeating the last channel
                            if was_chw:
                                padding = np.repeat(compressed_array[-1:], target_channels - current_channels, axis=0)
                                compressed_array = np.concatenate([compressed_array, padding], axis=0)
                            else:
                                padding = np.repeat(compressed_array[:, :, -1:], target_channels - current_channels, axis=2)
                                compressed_array = np.concatenate([compressed_array, padding], axis=2)
                        elif target_channels < current_channels:
                            # Crop channels
                            if was_chw:
                                compressed_array = compressed_array[:target_channels]
                            else:
                                compressed_array = compressed_array[:, :, :target_channels]
                    else:
                        compressed_array = compressed_array.reshape(original_shape)
                except Exception as e:
                    print(f"Cannot fix shape mismatch: {e}. Returning closest match.")
                    
        return compressed_array
    
    def get_compression_info(self):
        """Get information about current compression settings."""
        return {
            'level': self.quality_level,
            'quality': self.quality, 
            'description': self.description
        }
    
    @classmethod
    def get_all_levels(cls):
        """Get all available compression levels."""
        return cls.COMPRESSION_LEVELS.copy()

def estimate_compression_ratio(original_obs, compressed_obs):
    """
    Estimate compression ratio by comparing array sizes.
    This is an approximation since we don't have access to the actual JPEG byte size.
    """
    original_size = original_obs.nbytes
    
    # Simulate JPEG compression to get byte size
    if compressed_obs.dtype != np.uint8:
        if compressed_obs.max() <= 1.0:
            temp_obs = (compressed_obs * 255).astype(np.uint8)
        else:
            temp_obs = compressed_obs.astype(np.uint8)
    else:
        temp_obs = compressed_obs
        
    # Convert to PIL and get JPEG byte size
    try:
        if len(temp_obs.shape) == 3 and temp_obs.shape[0] == 3:  # CHW
            temp_obs = np.transpose(temp_obs, (1, 2, 0))
        elif len(temp_obs.shape) == 3 and temp_obs.shape[2] == 3:  # HWC
            pass
        else:  # Grayscale or single channel
            if len(temp_obs.shape) == 3:
                temp_obs = temp_obs[:, :, 0] if temp_obs.shape[2] == 1 else temp_obs[0, :, :]
                
        pil_image = Image.fromarray(temp_obs)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)  # Use high quality for size estimation
        compressed_size = buffer.tell()
        
        return original_size / compressed_size if compressed_size > 0 else 1.0
    except:
        return 1.0  # Fallback if estimation fails

if __name__ == "__main__":
    # Test compression with sample data
    print("Testing JPEG compression pipeline...")
    
    # Test with different observation formats
    test_cases = [
        ("RGB HWC", np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)),
        ("RGB CHW", np.random.randint(0, 255, (3, 84, 84), dtype=np.uint8)),
        ("Grayscale", np.random.randint(0, 255, (84, 84), dtype=np.uint8)),
        ("Float RGB", np.random.random((84, 84, 3)).astype(np.float32))
    ]
    
    for level_name in JPEGCompressor.COMPRESSION_LEVELS:
        print(f"\n=== Testing {level_name} compression ===")
        compressor = JPEGCompressor(level_name)
        print(f"Quality: {compressor.quality}")
        print(f"Description: {compressor.description}")
        
        for case_name, test_obs in test_cases:
            try:
                compressed = compressor.compress_observation(test_obs)
                ratio = estimate_compression_ratio(test_obs, compressed)
                
                print(f"  {case_name}: {test_obs.shape} -> {compressed.shape}, "
                      f"estimated ratio: {ratio:.2f}x")
                      
                # Verify shape preservation
                assert test_obs.shape == compressed.shape, f"Shape mismatch: {test_obs.shape} != {compressed.shape}"
                
            except Exception as e:
                print(f"  {case_name}: ERROR - {e}")
                
    print("\n✓ Compression pipeline test completed!")