#!/usr/bin/env python3
"""
Create plots and analysis for JPEG compression experiment results.
Generates bar charts and CSV exports from experiment results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_experiment_results(results_dir):
    """Load experiment results from JSON file."""
    results_path = Path(results_dir) / 'experiment_results.json'
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)

def create_dataframe(results):
    """Convert results to pandas DataFrame for analysis."""
    data = []
    
    for experiment_name, result in results.items():
        data.append({
            'experiment': experiment_name,
            'model': result['model_name'],
            'compression_level': result['compression_level'], 
            'mean_reward': result['mean_reward'],
            'std_reward': result['std_reward'],
            'mean_episode_length': result.get('mean_episode_length', 0),
            'compression_ratio': result.get('compression_ratio', 1.0),
            'noise_std': result.get('noise_std', 0.0),
            'num_episodes': result['num_episodes_completed']
        })
    
    return pd.DataFrame(data)

def create_compression_order():
    """Define the order of compression levels for plotting."""
    return ['none', 'minimal', 'low', 'medium', 'high', 'very_high']

def create_reward_bar_chart(df, output_dir):
    """Create bar chart of mean reward by compression level."""
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique models and compression levels
    models = df['model'].unique()
    compression_levels = create_compression_order()
    
    # Set up bar positions
    bar_width = 0.35
    x_pos = np.arange(len(compression_levels))
    
    # Create bars for each model
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        
        # Get data in correct order
        rewards = []
        errors = []
        for comp_level in compression_levels:
            data = model_data[model_data['compression_level'] == comp_level]
            if not data.empty:
                rewards.append(data['mean_reward'].iloc[0])
                errors.append(data['std_reward'].iloc[0])
            else:
                rewards.append(0)
                errors.append(0)
        
        # Create bars
        bars = ax.bar(x_pos + i * bar_width, rewards, bar_width,
                     label=model, yerr=errors, capsize=5, alpha=0.8)
        
        # Add value labels on bars
        for j, (bar, reward) in enumerate(zip(bars, rewards)):
            if reward > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + errors[j] + max(rewards)*0.01,
                       f'{reward:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Customize plot
    ax.set_xlabel('Compression Level', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Model Performance vs JPEG Compression Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + bar_width/2)
    ax.set_xticklabels([level.replace('_', '\n').title() for level in compression_levels])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plot_path = Path(output_dir) / 'plots' / 'reward_by_compression.png'
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Reward bar chart saved: {plot_path}")

def create_episode_length_bar_chart(df, output_dir):
    """Create bar chart of mean episode length by compression level."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = df['model'].unique()
    compression_levels = create_compression_order()
    
    bar_width = 0.35
    x_pos = np.arange(len(compression_levels))
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        
        lengths = []
        for comp_level in compression_levels:
            data = model_data[model_data['compression_level'] == comp_level]
            if not data.empty:
                lengths.append(data['mean_episode_length'].iloc[0])
            else:
                lengths.append(0)
        
        bars = ax.bar(x_pos + i * bar_width, lengths, bar_width,
                     label=model, alpha=0.8)
        
        # Add value labels
        for bar, length in zip(bars, lengths):
            if length > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(lengths)*0.01,
                       f'{length:.0f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Compression Level', fontsize=12)
    ax.set_ylabel('Mean Episode Length', fontsize=12)
    ax.set_title('Episode Length vs JPEG Compression Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + bar_width/2)
    ax.set_xticklabels([level.replace('_', '\n').title() for level in compression_levels])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(output_dir) / 'plots' / 'episode_length_by_compression.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Episode length bar chart saved: {plot_path}")

def create_compression_ratio_chart(df, output_dir):
    """Create chart showing compression ratio vs performance drop."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = df['model'].unique()
    
    for model in models:
        model_data = df[df['model'] == model].copy()
        
        # Calculate performance drop from baseline
        baseline_reward = model_data[model_data['compression_level'] == 'none']['mean_reward'].iloc[0]
        model_data['performance_drop'] = (baseline_reward - model_data['mean_reward']) / baseline_reward * 100
        
        # Plot
        ax.plot(model_data['compression_ratio'], model_data['performance_drop'], 
               marker='o', linewidth=2, markersize=8, label=model, alpha=0.8)
        
        # Add point labels
        for _, row in model_data.iterrows():
            ax.annotate(row['compression_level'], 
                       (row['compression_ratio'], row['performance_drop']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    ax.set_xlabel('Compression Ratio (x)', fontsize=12)
    ax.set_ylabel('Performance Drop (%)', fontsize=12)
    ax.set_title('Performance Drop vs Compression Ratio', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = Path(output_dir) / 'plots' / 'performance_vs_compression_ratio.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Compression ratio chart saved: {plot_path}")

def export_to_csv(df, output_dir):
    """Export results to CSV file."""
    # Reorder columns for better readability
    columns_order = ['experiment', 'model', 'compression_level', 'mean_reward', 'std_reward', 
                    'mean_episode_length', 'compression_ratio', 'noise_std', 'num_episodes']
    df_ordered = df[columns_order]
    
    # Sort by model and compression level
    compression_order = create_compression_order()
    df_ordered['compression_order'] = df_ordered['compression_level'].apply(
        lambda x: compression_order.index(x) if x in compression_order else 999
    )
    df_sorted = df_ordered.sort_values(['model', 'compression_order']).drop('compression_order', axis=1)
    
    # Save CSV
    csv_path = Path(output_dir) / 'experiment_results.csv'
    df_sorted.to_csv(csv_path, index=False)
    
    print(f"✅ CSV exported: {csv_path}")
    
    # Also create summary statistics
    summary_stats = create_summary_statistics(df)
    summary_path = Path(output_dir) / 'summary_statistics.csv'
    summary_stats.to_csv(summary_path, index=False)
    
    print(f"✅ Summary statistics CSV exported: {summary_path}")

def create_summary_statistics(df):
    """Create summary statistics table."""
    summary_data = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        baseline = model_data[model_data['compression_level'] == 'none']
        
        if baseline.empty:
            continue
            
        baseline_reward = baseline['mean_reward'].iloc[0]
        
        for _, row in model_data.iterrows():
            performance_drop = (baseline_reward - row['mean_reward']) / baseline_reward * 100
            
            summary_data.append({
                'model': model,
                'compression_level': row['compression_level'],
                'mean_reward': row['mean_reward'],
                'baseline_reward': baseline_reward,
                'performance_drop_percent': performance_drop,
                'compression_ratio': row['compression_ratio'],
                'noise_std': row['noise_std']
            })
    
    return pd.DataFrame(summary_data)

def analyze_and_plot_results(results_dir):
    """Main function to analyze results and create all plots."""
    try:
        print(f"🔍 Loading results from: {results_dir}")
        results = load_experiment_results(results_dir)
        
        print(f"📊 Processing {len(results)} experiments")
        df = create_dataframe(results)
        
        print(f"📈 Creating visualizations...")
        create_reward_bar_chart(df, results_dir)
        create_episode_length_bar_chart(df, results_dir)
        create_compression_ratio_chart(df, results_dir)
        
        print(f"💾 Exporting CSV files...")
        export_to_csv(df, results_dir)
        
        # Print summary
        print(f"\n📋 ANALYSIS SUMMARY")
        print(f"=" * 50)
        print(f"Models tested: {', '.join(df['model'].unique())}")
        print(f"Compression levels: {', '.join(df['compression_level'].unique())}")
        print(f"Total experiments: {len(df)}")
        
        # Show performance summary
        print(f"\n🎯 Performance Summary:")
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            baseline = model_data[model_data['compression_level'] == 'none']['mean_reward'].iloc[0]
            worst = model_data['mean_reward'].min()
            worst_level = model_data[model_data['mean_reward'] == worst]['compression_level'].iloc[0]
            drop = (baseline - worst) / baseline * 100
            
            print(f"  {model}:")
            print(f"    Baseline: {baseline:.1f} reward")
            print(f"    Worst: {worst:.1f} reward ({worst_level} compression)")
            print(f"    Max drop: {drop:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results_pretrained_realistic"
    
    success = analyze_and_plot_results(results_dir)
    exit(0 if success else 1)