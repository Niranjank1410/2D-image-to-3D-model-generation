## voxel based 3D reconstruction model which uses silhouette-based voxel carving approach ##

import numpy as np
import torch
import torch.nn as nn
import trimesh
import cv2
from scipy import ndimage


class SimpleVoxelReconstructor:
    """
    Simple voxel-based 3D reconstruction from 2D silhouettes
    This creates a 3D volume and carves it based on the 2D mask
    """
    
    def __init__(self, voxel_resolution=32, threshold=0.5):
        """
        Args:
            voxel_resolution: size of voxel grid (e.g., 32 means 32x32x32)
            threshold: threshold for considering a voxel occupied
        """
        self.voxel_resolution = voxel_resolution
        self.threshold = threshold
        
    def reconstruct(self, image, mask):
        """
        Reconstruct 3D mesh from 2D image and mask
        
        Args:
            image: torch tensor (3, H, W) or numpy array
            mask: torch tensor (1, H, W) or numpy array
        
        Returns:
            trimesh.Trimesh object
        """
        #converting to numpy if needed
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().cpu().numpy()
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        
        #resizing mask to voxel resolution
        mask_resized = cv2.resize(mask, (self.voxel_resolution, self.voxel_resolution))
        
        #creating 3D voxel grid using silhouette extrusion
        voxel_grid = np.zeros((self.voxel_resolution, self.voxel_resolution, self.voxel_resolution))
        
        #simple extrusion: if pixel is occupied in mask, fill corresponding voxel column
        for i in range(self.voxel_resolution):
            for j in range(self.voxel_resolution):
                if mask_resized[i, j] > self.threshold:
                    #filling the depth dimension with decreasing probability
                    #this creates a 3D shape from 2D silhouette
                    depth_profile = self._create_depth_profile(self.voxel_resolution)
                    voxel_grid[i, j, :] = depth_profile
        
        #applying smoothing to make it more realistic
        voxel_grid = ndimage.gaussian_filter(voxel_grid, sigma=1.0)
        
        #threshold to get binary voxels
        voxel_grid = (voxel_grid > self.threshold * 0.5).astype(np.float32)
        
        #converting voxel grid to mesh using marching cubes
        mesh = self._voxels_to_mesh(voxel_grid)
        
        return mesh
    
    def _create_depth_profile(self, resolution):
        """
        Create a depth profile for extrusion
        Makes the object have some depth variation
        """
        center = resolution // 2
        depth_profile = np.zeros(resolution)
        
        #creating a bell-shaped profile
        for k in range(resolution):
            dist_from_center = abs(k - center) / center
            depth_profile[k] = np.exp(-3 * dist_from_center**2)
        
        return depth_profile
    
    def _voxels_to_mesh(self, voxel_grid):
        """
        Convert voxel grid to mesh using marching cubes
        """
        from skimage import measure
        
        #using marching cubes to extract surface
        try:
            verts, faces, normals, values = measure.marching_cubes(
                voxel_grid, 
                level=0.5,
                spacing=(1.0, 1.0, 1.0)
            )
            
            #centering the mesh
            verts = verts - verts.mean(axis=0)
            
            #normalizing to unit sphere
            max_dist = np.max(np.linalg.norm(verts, axis=1))
            if max_dist > 0:
                verts = verts / max_dist
            
            #creating trimesh
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            
            #fixing mesh
            mesh.fill_holes()
            mesh.fix_normals()
            
        except Exception as e:
            print(f"Warning: Marching cubes failed: {e}")
            #returns a default sphere if marching cubes fails
            mesh = trimesh.creation.icosphere(radius=0.5)
        
        return mesh


class ImprovedVoxelReconstructor(SimpleVoxelReconstructor):
    """
    Improved version with better depth estimation
    Uses mask erosion and image features for better reconstruction
    """
    
    def __init__(self, voxel_resolution=32, threshold=0.5, use_features=True):
        super().__init__(voxel_resolution, threshold)
        self.use_features = use_features
    
    def reconstruct(self, image, mask):
        """
        Improved reconstruction using image features
        """
        #converting to numpy if needed
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().cpu().numpy()
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        
        #resizing
        mask_resized = cv2.resize(mask, (self.voxel_resolution, self.voxel_resolution))
        image_resized = cv2.resize(image, (self.voxel_resolution, self.voxel_resolution))
        
        #creating voxel grid
        voxel_grid = np.zeros((self.voxel_resolution, self.voxel_resolution, self.voxel_resolution))
        
        #using image intensity as depth cue if enabled
        if self.use_features and image_resized.max() > 0:
            intensity = np.mean(image_resized, axis=2)  # Convert to grayscale
        else:
            intensity = np.ones((self.voxel_resolution, self.voxel_resolution))
        
        #building 3D shape
        for i in range(self.voxel_resolution):
            for j in range(self.voxel_resolution):
                if mask_resized[i, j] > self.threshold:
                    #using intensity to modulate depth
                    depth_scale = 0.5 + 0.5 * intensity[i, j]  # Scale between 0.5 and 1.0
                    depth_profile = self._create_adaptive_depth_profile(
                        self.voxel_resolution, 
                        depth_scale
                    )
                    voxel_grid[i, j, :] = depth_profile
        
        #smoothening
        voxel_grid = ndimage.gaussian_filter(voxel_grid, sigma=1.2)
        
        #threshold
        voxel_grid = (voxel_grid > self.threshold * 0.4).astype(np.float32)
        
        #converting to mesh
        mesh = self._voxels_to_mesh(voxel_grid)
        
        return mesh
    
    def _create_adaptive_depth_profile(self, resolution, depth_scale):
        """
        Create depth profile with adaptive scaling
        """
        center = resolution // 2
        depth_profile = np.zeros(resolution)
        
        for k in range(resolution):
            dist_from_center = abs(k - center) / center
            depth_profile[k] = depth_scale * np.exp(-2.5 * dist_from_center**2)
        
        return depth_profile
