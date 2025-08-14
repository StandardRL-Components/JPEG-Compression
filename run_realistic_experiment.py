#!/usr/bin/env python3
"""
Run JPEG compression experiment on pre-trained realistic models.
"""

import os
import sys
import json
import time
from pathlib import Path
import logging
import numpy as np

# Set up environment for headless MuJoCo rendering
os.environ['MUJOCO_GL'] = 'egl'

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_realistic_experiment():
    """Run the complete JPEG compression experiment on pre-trained models."""
    
    try:
        from pretrained_realistic_config import (
            MODEL_CONFIGS, ENVIRONMENT_CONFIGS, COMPRESSION_CONFIGS,
            EXPERIMENT_CONFIG, ensure_directories
        )
        from pretrained_model_loader import PretrainedModelLoader
        from compression_utils import JPEGCompressor
        from simple_evaluation_runner import EvaluationRunner
        
        print("🚀 STARTING REALISTIC JPEG COMPRESSION EXPERIMENT")
        print("=" * 60)
        
        # Ensure directories exist
        ensure_directories()
        
        # Initialize components
        model_loader = PretrainedModelLoader()
        runner = EvaluationRunner(EXPERIMENT_CONFIG)
        
        all_results = {}
        experiment_count = 0
        total_experiments = len(MODEL_CONFIGS) * len(COMPRESSION_CONFIGS)
        
        print(f"📊 Experiment Overview:")
        print(f"   Models: {len(MODEL_CONFIGS)}")
        print(f"   Compression levels: {len(COMPRESSION_CONFIGS)}")
        print(f"   Total experiments: {total_experiments}")
        print(f"   Episodes per experiment: {EXPERIMENT_CONFIG['num_evaluation_episodes']}")
        print(f"   Total episodes: {total_experiments * EXPERIMENT_CONFIG['num_evaluation_episodes']}")
        print()
        
        # Run experiments for each model and compression level
        for model_name, model_config in MODEL_CONFIGS.items():
            print(f"🤖 Loading model: {model_name}")
            print(f"   Repository: {model_config['huggingface_repo']}")
            print(f"   Algorithm: {model_config['algorithm']}")
            print(f"   Environment: {model_config['environment']}")
            
            try:
                # Load pre-trained model
                model, env = model_loader.load_model(model_name)
                logger.info(f"✅ Model {model_name} loaded successfully")
                
                for compression_level in COMPRESSION_CONFIGS.keys():
                    experiment_count += 1
                    experiment_name = f"{model_name}_{compression_level}_compression"
                    
                    print(f"\n🔬 Running experiment {experiment_count}/{total_experiments}: {experiment_name}")
                    
                    try:
                        # Create compressor
                        compressor = JPEGCompressor(compression_level)
                        quality = COMPRESSION_CONFIGS[compression_level]['quality']
                        print(f"   Compression: {compression_level} (Quality: {quality}%)")
                        
                        # Run evaluation
                        results = runner.evaluate_model_with_compression(
                            model, env, compressor, experiment_name
                        )
                        
                        all_results[experiment_name] = results
                        
                        # Print results summary
                        mean_reward = results['mean_reward']
                        std_reward = results['std_reward']
                        compression_ratio = results.get('compression_ratio', 1.0)
                        
                        print(f"   ✅ Results: {mean_reward:.2f} ± {std_reward:.2f} reward")
                        print(f"      Compression ratio: {compression_ratio:.1f}x")
                        print(f"      Episodes completed: {results['num_episodes_completed']}")
                        
                        # Save intermediate results
                        runner.save_intermediate_results(all_results)
                        
                    except Exception as exp_e:
                        logger.error(f"❌ Experiment {experiment_name} failed: {exp_e}")
                        print(f"   ❌ Failed: {exp_e}")
                        continue
                
                # Close environment
                env.close()
                print(f"✅ Model {model_name} evaluation completed")
                
            except Exception as model_e:
                logger.error(f"❌ Model {model_name} failed to load: {model_e}")
                print(f"❌ Model {model_name} failed: {model_e}")
                continue
        
        print(f"\n📊 EXPERIMENT SUMMARY")
        print(f"=" * 40)
        print(f"Total experiments planned: {total_experiments}")
        print(f"Experiments completed: {len(all_results)}")
        print(f"Success rate: {len(all_results)/total_experiments*100:.1f}%")
        
        if all_results:
            # Save final results and create analysis
            results_dir = EXPERIMENT_CONFIG['results_dir']
            runner.save_experiment_results(all_results, results_dir)
            print(f"💾 Results saved to: {results_dir}")
            
            # Create plots and analysis
            try:
                from create_plots_and_analysis import analyze_and_plot_results
                analyze_and_plot_results(results_dir)
                logger.info("✅ Analysis and plots created")
                print(f"📈 Plots and analysis created in: {results_dir}/plots/")
                print(f"📊 CSV files exported to: {results_dir}/")
            except Exception as analysis_e:
                logger.warning(f"⚠️ Analysis failed: {analysis_e}")
                print(f"⚠️ Analysis failed: {analysis_e}")
            
            print(f"\n🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
            print(f"📋 View results in: {results_dir}/")
            print(f"📈 View plots in: {results_dir}/plots/")
            
        else:
            print(f"\n❌ NO EXPERIMENTS COMPLETED!")
            print(f"   Check logs for detailed error information")
        
        return all_results
        
    except Exception as e:
        logger.error(f"❌ Experiment setup failed: {e}")
        print(f"❌ Experiment setup failed: {e}")
        print(f"💡 Try running the test notebook (Workspace.ipynb) first")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    
    print("🧪 JPEG Compression Robustness Experiment")
    print("Testing pre-trained RL models with realistic robotic environments")
    print()
    
    start_time = time.time()
    results = run_realistic_experiment()
    end_time = time.time()
    
    if results:
        duration = end_time - start_time
        print(f"\n⏱️ Total runtime: {duration/3600:.1f} hours ({duration/60:.0f} minutes)")
        print(f"🎯 Experiments completed: {len(results)}")
        exit(0)
    else:
        print(f"\n❌ Experiment failed!")
        print(f"💡 Check the troubleshooting section in GUIDE.md")
        exit(1)