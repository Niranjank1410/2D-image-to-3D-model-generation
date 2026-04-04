## Eval metrics for 3d reconstruction which implements Chamfer Distance, F-Score, and IoU ##

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
import open3d as o3d


def mesh_to_point_cloud(mesh, n_points=10000):
    """
    Sample points from a mesh uniformly
    
    Args:
        mesh: trimesh.Trimesh object
        n_points: number of points to sample
    
    Returns:
        numpy array of shape (n_points, 3)
    """
    if isinstance(mesh, trimesh.Trimesh):
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
    elif isinstance(mesh, o3d.geometry.TriangleMesh):
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
        points = np.asarray(pcd.points)
    else:
        raise ValueError(f"Unsupported mesh type: {type(mesh)}")
    
    return points


def chamfer_distance(points1, points2):
    """
    Compute Chamfer Distance between two point clouds
    
    Args:
        points1: numpy array of shape (N, 3)
        points2: numpy array of shape (M, 3)
    
    Returns:
        chamfer_dist: scalar distance value
    """
    #building KD-trees for efficient nearest neighbor search
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    #for each point in points1, find nearest in points2
    dist1, _ = tree2.query(points1, k=1)
    
    #for each point in points2, find nearest in points1
    dist2, _ = tree1.query(points2, k=1)
    
    #Chamfer distance is the average of both directions
    chamfer_dist = (np.mean(dist1**2) + np.mean(dist2**2)) / 2.0
    
    return chamfer_dist


def f_score(points_pred, points_gt, threshold=0.01):
    """
    Compute F-Score (harmonic mean of precision and recall)
    
    Args:
        points_pred: predicted point cloud (N, 3)
        points_gt: ground truth point cloud (M, 3)
        threshold: distance threshold for considering a match
    
    Returns:
        f_score: F-score value (0 to 1, higher is better)
        precision: precision value
        recall: recall value
    """
    #building KD-trees
    tree_pred = cKDTree(points_pred)
    tree_gt = cKDTree(points_gt)
    
    # Precision: fraction of predicted points close to GT
    dist_pred_to_gt, _ = tree_gt.query(points_pred, k=1)
    precision = np.mean(dist_pred_to_gt < threshold)
    
    # Recall: fraction of GT points close to predictions
    dist_gt_to_pred, _ = tree_pred.query(points_gt, k=1)
    recall = np.mean(dist_gt_to_pred < threshold)
    
    # F-score (handle division by zero)
    if precision + recall == 0:
        f_score_val = 0.0
    else:
        f_score_val = 2 * (precision * recall) / (precision + recall)
    
    return f_score_val, precision, recall


def voxel_iou(mesh1, mesh2, voxel_resolution=32):
    """
    Compute IoU (Intersection over Union) using voxelization
    
    Args:
        mesh1: first mesh (trimesh or o3d)
        mesh2: second mesh (trimesh or o3d)
        voxel_resolution: resolution of voxel grid
    
    Returns:
        iou: IoU value (0 to 1, higher is better)
    """
    #connverting to trimesh if needed
    if isinstance(mesh1, o3d.geometry.TriangleMesh):
        vertices = np.asarray(mesh1.vertices)
        faces = np.asarray(mesh1.triangles)
        mesh1 = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    if isinstance(mesh2, o3d.geometry.TriangleMesh):
        vertices = np.asarray(mesh2.vertices)
        faces = np.asarray(mesh2.triangles)
        mesh2 = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    #normalizing both meshes to same bounds
    bounds1 = mesh1.bounds
    bounds2 = mesh2.bounds
    
    #using the union of bounds
    min_bound = np.minimum(bounds1[0], bounds2[0])
    max_bound = np.maximum(bounds1[1], bounds2[1])
    
    #creating voxel grids
    pitch = (max_bound - min_bound) / voxel_resolution
    
    try:
        voxels1 = mesh1.voxelized(pitch=np.max(pitch))
        voxels2 = mesh2.voxelized(pitch=np.max(pitch))
        
        #getting filled voxels as boolean arrays
        matrix1 = voxels1.matrix
        matrix2 = voxels2.matrix
        
        #ensuring same size (with padding if needed)
        max_shape = np.maximum(matrix1.shape, matrix2.shape)
        padded1 = np.zeros(max_shape, dtype=bool)
        padded2 = np.zeros(max_shape, dtype=bool)
        
        padded1[:matrix1.shape[0], :matrix1.shape[1], :matrix1.shape[2]] = matrix1
        padded2[:matrix2.shape[0], :matrix2.shape[1], :matrix2.shape[2]] = matrix2
        
        # computing IoU
        intersection = np.logical_and(padded1, padded2).sum()
        union = np.logical_or(padded1, padded2).sum()
        
        iou = intersection / union if union > 0 else 0.0
        
    except Exception as e:
        print(f"Warning: IoU computation failed: {e}")
        iou = 0.0
    
    return iou


def evaluate_reconstruction(pred_mesh, gt_mesh, n_points=10000, 
                           f_score_threshold=0.01, voxel_res=32):
    """
    Compute all metrics for a single reconstruction
    
    Args:
        pred_mesh: predicted mesh
        gt_mesh: ground truth mesh
        n_points: number of points to sample for Chamfer and F-Score
        f_score_threshold: threshold for F-Score computation
        voxel_res: voxel resolution for IoU
    
    Returns:
        dict with metrics: chamfer, f_score, precision, recall, iou
    """
    #sample point clouds
    points_pred = mesh_to_point_cloud(pred_mesh, n_points)
    points_gt = mesh_to_point_cloud(gt_mesh, n_points)
    
    #computing metrics
    chamfer = chamfer_distance(points_pred, points_gt)
    f_score_val, precision, recall = f_score(points_pred, points_gt, f_score_threshold)
    iou = voxel_iou(pred_mesh, gt_mesh, voxel_res)
    
    return {
        'chamfer_distance': float(chamfer),
        'f_score': float(f_score_val),
        'precision': float(precision),
        'recall': float(recall),
        'iou': float(iou)
    }


def test_metrics():
    """
    Test the metrics with dummy meshes
    """
    print("Testing evaluation metrics...")
    
    #creating two cube meshes
    mesh1 = trimesh.creation.box(extents=[1, 1, 1])
    mesh2 = trimesh.creation.box(extents=[1.1, 1.1, 1.1])
    
    #evaluating
    metrics = evaluate_reconstruction(mesh1, mesh2, n_points=1000, voxel_res=16)
    
    print("\nMetrics for two similar cubes:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    #testing with very different meshes
    mesh3 = trimesh.creation.icosphere(radius=0.5)
    metrics2 = evaluate_reconstruction(mesh1, mesh3, n_points=1000, voxel_res=16)
    
    print("\nMetrics for cube vs sphere:")
    for key, value in metrics2.items():
        print(f"  {key}: {value:.6f}")
    
    print("\nMetrics test completed successfully!")


if __name__ == '__main__':
    test_metrics()