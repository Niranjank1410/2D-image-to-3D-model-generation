"""
Visualize sample reconstructions
Shows input image, GT mesh, and predicted mesh
"""

import sys
import os
sys.path.insert(0, 'code')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
from pathlib import Path

from dataloader import Pix3DChairDataset
from reconstructor import ImprovedVoxelReconstructor


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


def visualize_sample(dataset, reconstructor, sample_idx, output_dir='results/samples'):
    """
    Visualize a single sample reconstruction
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    #getting sample
    sample = dataset[sample_idx]
    
    if sample['gt_mesh'] is None:
        print(f"Skipping sample {sample_idx}: No ground truth mesh")
        return False
    
    #reconstructing
    pred_mesh = reconstructor.reconstruct(sample['image'], sample['mask'])
    
    #creating figure
    fig = plt.figure(figsize=(20, 5))
    
    #plotting input image
    ax1 = fig.add_subplot(1, 4, 1)
    img = sample['image'].permute(1, 2, 0).numpy()
    ax1.imshow(img)
    ax1.set_title(f'Input Image (Sample {sample_idx})')
    ax1.axis('off')
    
    #plotting mask
    ax2 = fig.add_subplot(1, 4, 2)
    mask = sample['mask'].squeeze().numpy()
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Input Mask')
    ax2.axis('off')
    
    #plotting ground truth mesh
    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    plot_mesh_matplotlib(sample['gt_mesh'], ax3, 'Ground Truth Mesh', color='lightgreen')
    
    #plotting predicted mesh
    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    plot_mesh_matplotlib(pred_mesh, ax4, 'Predicted Mesh', color='lightcoral')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sample_{sample_idx:04d}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/sample_{sample_idx:04d}.png")
    plt.close()
    
    return True


def main():
    print("=" * 70)
    print("GENERATING SAMPLE VISUALIZATIONS")
    print("=" * 70)
    
    #loading dataset
    print("\nLoading dataset...")
    dataset = Pix3DChairDataset(
        root_dir='/home/niranjan/2D-image-to-3D-model-generation/data/pix3d'
    )
    
    #creating reconstructor (using best configuration)
    print("Creating reconstructor...")
    reconstructor = ImprovedVoxelReconstructor(
        voxel_resolution=64,  # this was the configuration from experiment tests
        threshold=0.5,
        use_features=True
    )
    
    #visualizing several samples
    print("\nGenerating visualizations for sample images...")
    
    #selecting diverse samples (first, middle, and some random ones)
    sample_indices = [0, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 3800]
    
    successful = 0
    for idx in sample_indices:
        if idx < len(dataset):
            if visualize_sample(dataset, reconstructor, idx):
                successful += 1
    
    print(f"\n Generated {successful} sample visualizations")
    print(f"Saved in: results/samples/")
    print("=" * 70)


if __name__ == '__main__':
    main()
