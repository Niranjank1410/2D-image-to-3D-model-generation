
"""
Visualization and Analysis Script
Creates plots and analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

#setting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def load_results(results_dir='results'):
    """
    Load all experiment results
    """
    with open(f'{results_dir}/comparison.json', 'r') as f:
        results = json.load(f)
    return results


def create_comparison_plot(results, output_dir='results/plots'):
    """
    Create bar plot comparing all experiments
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    #extracting data
    exp_names = [r['experiment_name'] for r in results]
    chamfer_means = [r['metrics']['chamfer_distance']['mean'] for r in results]
    chamfer_stds = [r['metrics']['chamfer_distance']['std'] for r in results]
    fscore_means = [r['metrics']['f_score']['mean'] for r in results]
    fscore_stds = [r['metrics']['f_score']['std'] for r in results]
    iou_means = [r['metrics']['iou']['mean'] for r in results]
    iou_stds = [r['metrics']['iou']['std'] for r in results]
    
    #creating figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    #Chamfer Distance (lower is better)
    axes[0].bar(range(len(exp_names)), chamfer_means, yerr=chamfer_stds, 
                capsize=5, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Experiment')
    axes[0].set_ylabel('Chamfer Distance')
    axes[0].set_title('Chamfer Distance (Lower is Better)')
    axes[0].set_xticks(range(len(exp_names)))
    axes[0].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    #F-Score (higher is better)
    axes[1].bar(range(len(exp_names)), fscore_means, yerr=fscore_stds,
                capsize=5, alpha=0.7, color='forestgreen')
    axes[1].set_xlabel('Experiment')
    axes[1].set_ylabel('F-Score')
    axes[1].set_title('F-Score (Higher is Better)')
    axes[1].set_xticks(range(len(exp_names)))
    axes[1].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    #IoU (higher is better)
    axes[2].bar(range(len(exp_names)), iou_means, yerr=iou_stds,
                capsize=5, alpha=0.7, color='coral')
    axes[2].set_xlabel('Experiment')
    axes[2].set_ylabel('IoU')
    axes[2].set_title('IoU (Higher is Better)')
    axes[2].set_xticks(range(len(exp_names)))
    axes[2].set_xticklabels(exp_names, rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/metrics_comparison.png")
    plt.close()


def create_hyperparameter_analysis(results, output_dir='results/plots'):
    """
    Analyze impact of hyperparameters
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    #extracting baseline and tuning results
    baseline = None
    voxel_res_64 = None
    threshold_03 = None
    threshold_07 = None
    
    for r in results:
        if r['experiment_name'] == 'baseline_improved':
            baseline = r
        elif r['experiment_name'] == 'tuning_voxel_res_64':
            voxel_res_64 = r
        elif r['experiment_name'] == 'tuning_threshold_0.3':
            threshold_03 = r
        elif r['experiment_name'] == 'tuning_threshold_0.7':
            threshold_07 = r
    
    #Voxel Resolution Impact
    if baseline and voxel_res_64:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['chamfer_distance', 'f_score', 'iou']
        titles = ['Chamfer Distance', 'F-Score', 'IoU']
        colors = ['steelblue', 'forestgreen', 'coral']
        
        for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            values = [
                baseline['metrics'][metric]['mean'],
                voxel_res_64['metrics'][metric]['mean']
            ]
            errors = [
                baseline['metrics'][metric]['std'],
                voxel_res_64['metrics'][metric]['std']
            ]
            
            axes[i].bar(['Res=32', 'Res=64'], values, yerr=errors, 
                       capsize=5, alpha=0.7, color=color)
            axes[i].set_ylabel(title)
            axes[i].set_title(f'Impact of Voxel Resolution on {title}')
            axes[i].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/voxel_resolution_impact.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/voxel_resolution_impact.png")
        plt.close()
    
    #Threshold Impact
    if baseline and threshold_03 and threshold_07:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            values = [
                threshold_03['metrics'][metric]['mean'],
                baseline['metrics'][metric]['mean'],
                threshold_07['metrics'][metric]['mean']
            ]
            errors = [
                threshold_03['metrics'][metric]['std'],
                baseline['metrics'][metric]['std'],
                threshold_07['metrics'][metric]['std']
            ]
            
            axes[i].bar(['T=0.3', 'T=0.5', 'T=0.7'], values, yerr=errors,
                       capsize=5, alpha=0.7, color=color)
            axes[i].set_ylabel(title)
            axes[i].set_title(f'Impact of Threshold on {title}')
            axes[i].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/threshold_impact.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/threshold_impact.png")
        plt.close()


def create_distribution_plots(results, output_dir='results/plots'):
    """
    Create distribution plots for each metric
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for result in results:
        exp_name = result['experiment_name']
        per_sample = result['per_sample_metrics']
        
        #extracting metrics
        chamfer = [s['chamfer_distance'] for s in per_sample]
        fscore = [s['f_score'] for s in per_sample]
        iou = [s['iou'] for s in per_sample]
        
        #creating distribution plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        #filtering outliers for better visualization (using IQR method)
        def filter_outliers(data):
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return [x for x in data if lower <= x <= upper]
        
        chamfer_filtered = filter_outliers(chamfer)
        
        axes[0].hist(chamfer_filtered, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Chamfer Distance')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Chamfer Distance Distribution\n{exp_name}')
        axes[0].axvline(np.median(chamfer), color='red', linestyle='--', label='Median')
        axes[0].legend()
        
        axes[1].hist(fscore, bins=50, alpha=0.7, color='forestgreen', edgecolor='black')
        axes[1].set_xlabel('F-Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'F-Score Distribution\n{exp_name}')
        axes[1].axvline(np.median(fscore), color='red', linestyle='--', label='Median')
        axes[1].legend()
        
        axes[2].hist(iou, bins=50, alpha=0.7, color='coral', edgecolor='black')
        axes[2].set_xlabel('IoU')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title(f'IoU Distribution\n{exp_name}')
        axes[2].axvline(np.median(iou), color='red', linestyle='--', label='Median')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/distribution_{exp_name}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir}/distribution_{exp_name}.png")
        plt.close()


def generate_summary_table(results, output_dir='results'):
    """
    Generate a summary table in markdown format
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(f'{output_dir}/summary_table.md', 'w') as f:
        f.write("# 3D Reconstruction Evaluation Results\n\n")
        f.write("## Summary Table\n\n")
        f.write("| Experiment | Chamfer ↓ | F-Score ↑ | IoU ↑ | Successful |\n")
        f.write("|------------|-----------|-----------|-------|------------|\n")
        
        for r in results:
            exp_name = r['experiment_name']
            chamfer = r['metrics']['chamfer_distance']['mean']
            chamfer_std = r['metrics']['chamfer_distance']['std']
            fscore = r['metrics']['f_score']['mean']
            fscore_std = r['metrics']['f_score']['std']
            iou = r['metrics']['iou']['mean']
            iou_std = r['metrics']['iou']['std']
            success = r['num_successful']
            
            f.write(f"| {exp_name} | {chamfer:.4f}±{chamfer_std:.4f} | "
                   f"{fscore:.4f}±{fscore_std:.4f} | {iou:.4f}±{iou_std:.4f} | "
                   f"{success}/3839 |\n")
        
        f.write("\n## Key Findings\n\n")
        
        #finding best configuration for each metric
        best_chamfer = min(results, key=lambda x: x['metrics']['chamfer_distance']['mean'])
        best_fscore = max(results, key=lambda x: x['metrics']['f_score']['mean'])
        best_iou = max(results, key=lambda x: x['metrics']['iou']['mean'])
        
        f.write(f"- **Best Chamfer Distance**: {best_chamfer['experiment_name']} "
               f"({best_chamfer['metrics']['chamfer_distance']['mean']:.6f})\n")
        f.write(f"- **Best F-Score**: {best_fscore['experiment_name']} "
               f"({best_fscore['metrics']['f_score']['mean']:.6f})\n")
        f.write(f"- **Best IoU**: {best_iou['experiment_name']} "
               f"({best_iou['metrics']['iou']['mean']:.6f})\n")
        
        f.write("\n## Hyperparameter Analysis\n\n")
        f.write("### Voxel Resolution\n")
        f.write("- Increasing voxel resolution from 32 to 64 improved F-Score\n")
        f.write("- Trade-off: Higher resolution increases computation time\n\n")
        
        f.write("### Threshold Value\n")
        f.write("- Threshold affects the density of reconstructed voxels\n")
        f.write("- Optimal threshold depends on the target metric\n")
    
    print(f"Saved: {output_dir}/summary_table.md")


def main():
    print("=" * 70)
    print("GENERATING VISUALIZATIONS AND ANALYSIS")
    print("=" * 70)
    
    # Load results
    print("\nLoading results...")
    results = load_results()
    print(f"Loaded {len(results)} experiments")
    
    # Create visualizations
    print("\nCreating comparison plots...")
    create_comparison_plot(results)
    
    print("\nCreating hyperparameter analysis...")
    create_hyperparameter_analysis(results)
    
    print("\nCreating distribution plots...")
    create_distribution_plots(results)
    
    print("\nGenerating summary table...")
    generate_summary_table(results)
    
    print("\n" + "=" * 70)
    print("ALL VISUALIZATIONS COMPLETED!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - results/plots/metrics_comparison.png")
    print("  - results/plots/voxel_resolution_impact.png")
    print("  - results/plots/threshold_impact.png")
    print("  - results/plots/distribution_*.png (for each experiment)")
    print("  - results/summary_table.md")
    print("\n")


if __name__ == '__main__':
    main()
