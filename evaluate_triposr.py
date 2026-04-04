"""
Evaluate TripoSR results using chosen metrics
"""

import sys
import os

#adding the 3d_reconstruction/code directory to Python path
metrics_path = os.path.abspath('3d_reconstruction/code')
sys.path.insert(0, metrics_path)

import json
import trimesh
import numpy as np
from tqdm import tqdm
from metrics import evaluate_reconstruction

def evaluate_triposr_results(mapping_file, output_file):
    """
    Evaluate TripoSR predictions against ground truth
    """
    # Load mapping
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    results = mapping['results']
    print(f"Evaluating {len(results)} TripoSR predictions...")
    
    all_metrics = []
    failed = []
    
    for item in tqdm(results, desc="Evaluating"):
        try:
            #loading predicted mesh
            pred_mesh = trimesh.load(item['triposr_mesh'], force='mesh', process=False)
            
            #loading ground truth mesh
            gt_path = '/home/niranjan/2D-image-to-3D-model-generation/data/pix3d/' + item['model_path']
            gt_mesh = trimesh.load(gt_path, force='mesh', process=False)
            
            if isinstance(gt_mesh, trimesh.Scene):
                gt_mesh = list(gt_mesh.geometry.values())[0]
            
            # Evaluate
            metrics = evaluate_reconstruction(
                pred_mesh, 
                gt_mesh,
                n_points=10000,
                f_score_threshold=0.01,
                voxel_res=32
            )
            
            metrics['sample_idx'] = item['sample_idx']
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"\nError evaluating sample {item['sample_idx']}: {e}")
            failed.append(item['sample_idx'])
    
    #computing statistics
    if len(all_metrics) == 0:
        print("ERROR: No samples evaluated successfully!")
        return None
    
    chamfer = [m['chamfer_distance'] for m in all_metrics]
    fscore = [m['f_score'] for m in all_metrics]
    iou = [m['iou'] for m in all_metrics]
    
    evaluation_results = {
        'method': 'TripoSR',
        'num_evaluated': len(all_metrics),
        'num_failed': len(failed),
        'metrics': {
            'chamfer_distance': {
                'mean': float(np.mean(chamfer)),
                'std': float(np.std(chamfer)),
                'median': float(np.median(chamfer)),
                'min': float(np.min(chamfer)),
                'max': float(np.max(chamfer))
            },
            'f_score': {
                'mean': float(np.mean(fscore)),
                'std': float(np.std(fscore)),
                'median': float(np.median(fscore)),
                'min': float(np.min(fscore)),
                'max': float(np.max(fscore))
            },
            'iou': {
                'mean': float(np.mean(iou)),
                'std': float(np.std(iou)),
                'median': float(np.median(iou)),
                'min': float(np.min(iou)),
                'max': float(np.max(iou))
            }
        },
        'per_sample_metrics': all_metrics
    }
    
    #saving results
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TRIPOSR EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Evaluated: {len(all_metrics)} samples")
    print(f"Failed: {len(failed)} samples")
    print("\nMetrics (mean ± std):")
    print(f"  Chamfer Distance: {evaluation_results['metrics']['chamfer_distance']['mean']:.6f} ± {evaluation_results['metrics']['chamfer_distance']['std']:.6f}")
    print(f"  F-Score:          {evaluation_results['metrics']['f_score']['mean']:.6f} ± {evaluation_results['metrics']['f_score']['std']:.6f}")
    print(f"  IoU:              {evaluation_results['metrics']['iou']['mean']:.6f} ± {evaluation_results['metrics']['iou']['std']:.6f}")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_file}")
    
    return evaluation_results


if __name__ == '__main__':
    mapping_file = 'triposr_results/triposr_mapping.json'
    output_file = 'triposr_results/triposr_evaluation.json'
    
    evaluate_triposr_results(mapping_file, output_file)