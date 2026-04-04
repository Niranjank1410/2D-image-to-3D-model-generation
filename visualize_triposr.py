"""
Visualize TripoSR reconstruction samples
Shows input image, GT mesh, and TripoSR predicted mesh side-by-side
"""

import sys
import os
sys.path.insert(0, '3d_reconstruction/code')

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from pathlib import Path
from PIL import Image


def plot_mesh_matplotlib(mesh, ax, title='', color='lightblue'):
    """
    Plot a trimesh mesh using matplotlib
    """
    vertices = mesh.vertices
    faces = mesh.faces
    
    #plotting the mesh
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2],
                    color=color, alpha=0.8, edgecolor='none', shade=True)
    
    #setting equal aspect ratio
    max_range = np.array([
        vertices[:, 0].max() - vertices[:, 0].min(),
        vertices[:, 1].max() - vertices[:, 1].min(),
        vertices[:, 2].max() - vertices[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.view_init(elev=20, azim=45)


def visualize_triposr_sample(mapping_file, sample_indices, output_dir='triposr_results/visualizations'):
    """
    Visualize TripoSR samples
    
    Args:
        mapping_file: Path to triposr_mapping.json
        sample_indices: List of sample indices to visualize
        output_dir: Where to save visualizations
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    #loading mappings
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    results = mapping['results']
    
    print(f"Visualizing {len(sample_indices)} TripoSR samples...")
    
    for idx in sample_indices:
        #finding the result for this sample
        sample_result = None
        for r in results:
            if r['sample_idx'] == idx:
                sample_result = r
                break
        
        if sample_result is None:
            print(f"Sample {idx} not found in results, skipping...")
            continue
        
        try:
            #loading input image
            img = Image.open(sample_result['img_path'])
            
            #loading ground truth mesh
            gt_path = os.path.join('/home/niranjan/2D-image-to-3D-model-generation/data/pix3d/' + sample_result['model_path'])
            gt_mesh = trimesh.load(gt_path, force='mesh', process=False)
            if isinstance(gt_mesh, trimesh.Scene):
                gt_mesh = list(gt_mesh.geometry.values())[0]
            
            #loading TripoSR predicted mesh
            pred_mesh = trimesh.load(sample_result['triposr_mesh'], force='mesh', process=False)
            
            #creating figure
            fig = plt.figure(figsize=(18, 6))
            
            #plotting input image
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(img)
            ax1.set_title(f'Input Image (Sample {idx})', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            #plotting ground truth mesh
            ax2 = fig.add_subplot(1, 3, 2, projection='3d')
            plot_mesh_matplotlib(gt_mesh, ax2, 'Ground Truth Mesh', color='lightgreen')
            
            #plotting TripoSR predicted mesh
            ax3 = fig.add_subplot(1, 3, 3, projection='3d')
            plot_mesh_matplotlib(pred_mesh, ax3, 'TripoSR Prediction', color='lightcoral')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/triposr_sample_{idx:04d}.png', dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_dir}/triposr_sample_{idx:04d}.png")
            plt.close()
            
        except Exception as e:
            print(f"Error visualizing sample {idx}: {e}")
            continue


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping_file', type=str, 
                       default='triposr_results/triposr_mapping.json')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    #loading mapping to see how many samples are present
    with open(args.mapping_file, 'r') as f:
        mapping = json.load(f)
    
    num_available = len(mapping['results'])
    print(f"Found {num_available} TripoSR results")
    
    #selecting diverse samples to visualize
    if num_available <= args.num_samples:
        sample_indices = [r['sample_idx'] for r in mapping['results']]
    else:
        # Select evenly spaced samples
        step = num_available // args.num_samples
        sample_indices = [mapping['results'][i*step]['sample_idx'] 
                         for i in range(args.num_samples)]
    
    print(f"Visualizing samples: {sample_indices}")
    
    visualize_triposr_sample(args.mapping_file, sample_indices)
    
    print("\nVisualization complete!")
    print(f"Saved to: triposr_results/visualizations/")


if __name__ == '__main__':
    main()