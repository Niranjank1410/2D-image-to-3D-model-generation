"""
Visualize TripoSR Evaluation Results
Creates plots comparing TripoSR performance
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_results():
    """Load both baseline and TripoSR results"""
    #loading baseline
    with open('3d_reconstruction/results/baseline_improved.json', 'r') as f:
        baseline = json.load(f)
    
    #loading TripoSR
    with open('triposr_results/triposr_evaluation.json', 'r') as f:
        triposr = json.load(f)
    
    return baseline, triposr


def create_comparison_bar_plot(baseline, triposr, output_dir='triposr_results/plots'):
    """
    Create bar plot comparing baseline vs TripoSR
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    metrics = ['chamfer_distance', 'f_score', 'iou']
    metric_names = ['Chamfer Distance\n(lower better)', 'F-Score\n(higher better)', 'IoU\n(higher better)']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        baseline_mean = baseline['metrics'][metric]['mean']
        baseline_std = baseline['metrics'][metric]['std']
        triposr_mean = triposr['metrics'][metric]['mean']
        triposr_std = triposr['metrics'][metric]['std']
        
        #creating bars
        x = [0, 1]
        means = [baseline_mean, triposr_mean]
        stds = [baseline_std, triposr_std]
        labels = ['Your Baseline\n(Voxel)', 'TripoSR\n(Learning-Based)']
        colors = ['steelblue', 'coral']
        
        axes[i].bar(x, means, yerr=stds, capsize=5, alpha=0.8, color=colors, edgecolor='black', linewidth=1.5)
        axes[i].set_ylabel(name.split('\n')[0], fontsize=12, fontweight='bold')
        axes[i].set_title(name, fontsize=13, fontweight='bold')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(labels, fontsize=11)
        axes[i].grid(axis='y', alpha=0.3)
        
        #adding improvement annotation
        if metric == 'chamfer_distance':
            improvement = baseline_mean / triposr_mean
            axes[i].text(0.5, max(means) * 0.8, f'{improvement:.1f}× better', 
                        ha='center', fontsize=14, fontweight='bold', 
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        else:
            improvement = triposr_mean / baseline_mean
            axes[i].text(0.5, max(means) * 0.5, f'{improvement:.1f}× better', 
                        ha='center', fontsize=14, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.suptitle('Baseline vs TripoSR Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/triposr_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/triposr_comparison.png")
    plt.close()


def create_distribution_comparison(baseline, triposr, output_dir='triposr_results/plots'):
    """
    Create distribution plots comparing baseline vs TripoSR
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    #getting per-sample metrics
    baseline_samples = baseline['per_sample_metrics']
    triposr_samples = triposr['per_sample_metrics']
    
    #extracting metrics
    baseline_chamfer = [s['chamfer_distance'] for s in baseline_samples]
    baseline_fscore = [s['f_score'] for s in baseline_samples]
    baseline_iou = [s['iou'] for s in baseline_samples]
    
    triposr_chamfer = [s['chamfer_distance'] for s in triposr_samples]
    triposr_fscore = [s['f_score'] for s in triposr_samples]
    triposr_iou = [s['iou'] for s in triposr_samples]
    
    #creating figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Chamfer Distance
    axes[0].hist(baseline_chamfer[:1000], bins=50, alpha=0.6, label='Baseline', color='steelblue', edgecolor='black')
    axes[0].hist(triposr_chamfer, bins=50, alpha=0.6, label='TripoSR', color='coral', edgecolor='black')
    axes[0].set_xlabel('Chamfer Distance', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Chamfer Distance Distribution (Lower is Better)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].set_xlim(0, 0.2)  # Focus on lower range where TripoSR is
    axes[0].grid(axis='y', alpha=0.3)
    
    # F-Score
    axes[1].hist(baseline_fscore[:1000], bins=50, alpha=0.6, label='Baseline', color='steelblue', edgecolor='black')
    axes[1].hist(triposr_fscore, bins=50, alpha=0.6, label='TripoSR', color='coral', edgecolor='black')
    axes[1].set_xlabel('F-Score', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('F-Score Distribution (Higher is Better)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)
    
    # IoU
    axes[2].hist(baseline_iou[:1000], bins=50, alpha=0.6, label='Baseline', color='steelblue', edgecolor='black')
    axes[2].hist(triposr_iou, bins=50, alpha=0.6, label='TripoSR', color='coral', edgecolor='black')
    axes[2].set_xlabel('IoU', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('IoU Distribution (Higher is Better)', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Metric Distributions: Baseline vs TripoSR', fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/triposr_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/triposr_distributions.png")
    plt.close()


def create_summary_table(baseline, triposr, output_dir='triposr_results'):
    """
    Create markdown summary table
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(f'{output_dir}/comparison_summary.md', 'w') as f:
        f.write("# TripoSR Evaluation Summary\n\n")
        f.write("## Performance Comparison\n\n")
        f.write("| Method | Samples | Chamfer Distance ↓ | F-Score ↑ | IoU ↑ |\n")
        f.write("|--------|---------|-------------------|-----------|-------|\n")
        
        # Baseline
        f.write(f"| Baseline (Voxel) | {baseline['num_successful']} | "
               f"{baseline['metrics']['chamfer_distance']['mean']:.6f} ± {baseline['metrics']['chamfer_distance']['std']:.6f} | "
               f"{baseline['metrics']['f_score']['mean']:.6f} ± {baseline['metrics']['f_score']['std']:.6f} | "
               f"{baseline['metrics']['iou']['mean']:.6f} ± {baseline['metrics']['iou']['std']:.6f} |\n")
        
        # TripoSR
        f.write(f"| TripoSR (Learning) | {triposr['num_evaluated']} | "
               f"{triposr['metrics']['chamfer_distance']['mean']:.6f} ± {triposr['metrics']['chamfer_distance']['std']:.6f} | "
               f"{triposr['metrics']['f_score']['mean']:.6f} ± {triposr['metrics']['f_score']['std']:.6f} | "
               f"{triposr['metrics']['iou']['mean']:.6f} ± {triposr['metrics']['iou']['std']:.6f} |\n")
        
        # Improvement
        chamfer_improvement = baseline['metrics']['chamfer_distance']['mean'] / triposr['metrics']['chamfer_distance']['mean']
        fscore_improvement = triposr['metrics']['f_score']['mean'] / baseline['metrics']['f_score']['mean']
        iou_improvement = triposr['metrics']['iou']['mean'] / baseline['metrics']['iou']['mean']
        
        f.write(f"| **Improvement** | - | "
               f"**{chamfer_improvement:.1f}×** | **{fscore_improvement:.1f}×** | **{iou_improvement:.1f}×** |\n\n")
        
        f.write("## Key Findings\n\n")
        f.write(f"- **TripoSR achieves {chamfer_improvement:.1f}× better Chamfer Distance** than the geometric baseline\n")
        f.write(f"- **TripoSR achieves {fscore_improvement:.1f}× better F-Score** than the geometric baseline\n")
        f.write(f"- **TripoSR achieves {iou_improvement:.1f}× better IoU** than the geometric baseline\n\n")
    
    print(f"Saved: {output_dir}/comparison_summary.md")


def main():
    print("=" * 70)
    print("GENERATING TRIPOSR EVALUATION VISUALIZATIONS")
    print("=" * 70)
    
    #loading results
    print("\nLoading results...")
    baseline, triposr = load_results()
    
    print(f"Baseline: {baseline['num_successful']} samples")
    print(f"TripoSR: {triposr['num_evaluated']} samples")
    
    #creating visualizations
    print("\nCreating comparison plots...")
    create_comparison_bar_plot(baseline, triposr)
    
    print("\nCreating distribution plots...")
    create_distribution_comparison(baseline, triposr)
    
    print("\nGenerating summary table...")
    create_summary_table(baseline, triposr)
    
    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS COMPLETED!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - triposr_results/plots/triposr_comparison.png")
    print("  - triposr_results/plots/triposr_distributions.png")
    print("  - triposr_results/comparison_summary.md")
    print("\n")


if __name__ == '__main__':
    main()