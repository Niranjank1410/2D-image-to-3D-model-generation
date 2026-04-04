## Main pipeline for 3D reconstruction evaluation which runs the baseline + hyperparameter tuning experiments ##

import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm
import argparse
from datetime import datetime

#adding code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from dataloader import Pix3DChairDataset
from reconstructor import SimpleVoxelReconstructor, ImprovedVoxelReconstructor
from metrics import evaluate_reconstruction

class ExperimentRunner:
    """
    Runs reconstruction experiments with different hyperparameters
    """

    def __init__(self, dataset, results_dir='results'):
        self.dataset = dataset
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def run_experiment(self, config, experiment_name, num_samples=None):
        """
        Runs a single experiment with given configurations
        
        Args:
            config: dict with hyperparameters
            experiment_name: name of the experiment
            num_samples: number of samples to evaluate (None = All)
        
        Returns:
            dict with results
        """
        
        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment_name}")
        print(f"{'='*60}")
        print(f"Configuration: {json.dumps(config, indent=2)}")

        #creating reconstructor based on config
        if config['model_type'] == 'simple':
            reconstructor = SimpleVoxelReconstructor(
                voxel_resolution = config['voxel_resolution'],
                threshold = config['threshold']
            )
        else:
            reconstructor = ImprovedVoxelReconstructor(
                voxel_resolution = config['voxel_resolution'],
                threshold = config['threshold'],
                use_features = config.get('use_features', True)
            )
        
        #determining number of samples
        if num_samples is None:
            num_samples = len(self.dataset)
        else:
            num_samples = min(num_samples, len(self.dataset))
        
        print(f"\nEvaluating on {num_samples} samples")

        #running evaluation
        all_metrics = []
        failed_samples = []

        for i in tqdm(range(num_samples), desc="Processing"):
            try:
                #getting sample
                sample = self.dataset[i]

                #skipping if no ground truth mesh
                if sample['gt_mesh'] is None:
                    failed_samples.append(i)
                    continue

                #reconstructing
                pred_mesh = reconstructor.reconstruct(sample['image'], sample['mask'])

                #evaluation
                metrics = evaluate_reconstruction(
                    pred_mesh,
                    sample['gt_mesh'],
                    n_points = config.get('n_points', 10000),
                    f_score_threshold = config.get('f_score_threshold', 0.01),
                    voxel_res = config.get('iou_voxel_res', 32)
                )

                metrics['sample_idx'] = i
                all_metrics.append(metrics)

            except Exception as e:
                print(f"\nError processing sample {i}: {e}")
                failed_samples.append(i)
                continue

        #computing statistics
        if len(all_metrics) == 0:
            print("ERROR: No samples were successfully processed!")
            return None
        
        results = self._compute_statistics(all_metrics, config, experiment_name)
        results['failed_samples'] = failed_samples
        results['num_successful'] = len(all_metrics)
        results['num_failed'] = len(failed_samples)

        #saving results
        self._save_results(results, experiment_name)

        #printing summary
        self._print_summary(results)

        return results

    def _compute_statistics(self, all_metrics, config, experiment_name):
        """
        Computes mean and std of all metrics
        """

        #extracting metric arrays
        chamfer = [m['chamfer_distance'] for m in all_metrics]
        f_score = [m['f_score'] for m in all_metrics]
        iou = [m['iou'] for m in all_metrics]
        precision = [m['precision'] for m in all_metrics]
        recall = [m['recall'] for m in all_metrics]

        results = {
            'experiment_name': experiment_name,
            'config': config,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'metrics': {
                'chamfer_distance': {
                    'mean': float(np.mean(chamfer)),
                    'std': float(np.std(chamfer)),
                    'min': float(np.min(chamfer)),
                    'max': float(np.max(chamfer)),
                    'median': float(np.median(chamfer))
                },
                'f_score': {
                    'mean': float(np.mean(f_score)),
                    'std': float(np.std(f_score)),
                    'min': float(np.min(f_score)),
                    'max': float(np.max(f_score)),
                    'median': float(np.median(f_score))
                },
                'iou': {
                    'mean': float(np.mean(iou)),
                    'std': float(np.std(iou)),
                    'min': float(np.min(iou)),
                    'max': float(np.max(iou)),
                    'median': float(np.median(iou))
                },
                'precision': {
                    'mean': float(np.mean(precision)),
                    'std': float(np.std(precision))
                },
                'recall': {
                    'mean': float(np.mean(recall)),
                    'std': float(np.std(recall))
                }
            },
            'per_sample_metrics': all_metrics
        }

        return results
    
    def _save_results(self, results, experiment_name):
        """
        Saves results to JSON file
        """

        #saves full results
        filepath = os.path.join(self.results_dir, f"{experiment_name}.json")
        with open(filepath, 'w') as f:
            json.dump(results, f, indent = 2)

        print(f"\nResults saved to: {filepath}")

    def _print_summary(self, results):
        """
        Prints summary of the results
        """
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY: {results['experiment_name']}")
        print(f"{'='*60}")
        print(f"Successful samples: {results['num_successful']}")
        print(f"Failed samples: {results['num_failed']}")
        print(f"\nMetrics (mean ± std):")
        print(f"  Chamfer Distance: {results['metrics']['chamfer_distance']['mean']:.6f} ± {results['metrics']['chamfer_distance']['std']:.6f}")
        print(f"  F-Score: {results['metrics']['f_score']['mean']:.6f} ± {results['metrics']['f_score']['std']:.6f}")
        print(f"  IoU: {results['metrics']['iou']['mean']:.6f} ± {results['metrics']['iou']['std']:.6f}")
        print(f"  Precision: {results['metrics']['precision']['mean']:.6f} ± {results['metrics']['precision']['std']:.6f}")
        print(f"  Recall: {results['metrics']['recall']['mean']:.6f} ± {results['metrics']['recall']['std']:.6f}")
        print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description = 'Run 3D Reconstruction Experiments')

    parser.add_argument('--data_root', type = str, default = '/home/niranjan/2D-image-to-3D-model-generation/data/pix3d', help = 'Path to Pix3D dataset')
    parser.add_argument('--results_dir', type = str, default = 'results', help = 'Directory to dave results')
    parser.add_argument('--num_samples', type = int, default = 100, help = 'Number of samples to evaluate (default: 100, use -1 for all)')
    parser.add_argument('--run_all', action='store_true', help = 'Run allexperiments (baseline + hyperparameter tuning)')

    args = parser.parse_args()

    #loading dataset
    print("Loading Pix3D dataset...")
    dataset = Pix3DChairDataset(root_dir = args.data_root)

    #creating experiment runner
    runner = ExperimentRunner(dataset, results_dir = args.results_dir)

    #determining number of samples
    num_samples = None if args.num_samples == -1 else args.num_samples

    #defining experiments
    experiments = []

    #Baseline 1: Simple reconstructor with default parameters
    experiments.append({
        'name': 'baseline_simple',
        'config': {
            'model_type': 'simple',
            'voxel_resolution': 32,
            'threshold': 0.5,
            'n_points': 10000,
            'f_score_threshold': 0.01,
            'iou_voxel_res': 32
        }
    })

    #Baseline 2: Improved reconstructor with default parameters
    experiments.append({
        'name': 'baseline_improved',
        'config': {
            'model_type': 'improved',
            'voxel_resolution': 32,
            'threshold': 0.5,
            'use_features': True,
            'n_points': 10000,
            'f_score_threshold': 0.01,
            'iou_voxel_res': 32
        }
    })

    if args.run_all:
        #hyperparameter tuning: Voxel resolution
        experiments.append({
            'name': 'tuning_voxel_res_64',
            'config': {
                'model_type': 'improved',
                'voxel_resolution': 64,
                'threshold': 0.5,
                'use_features': True,
                'n_points': 10000,
                'f_score_threshold': 0.01,
                'iou_voxel_res': 64
            }
        })

        #hyperparameter tuning: Threshold
        experiments.append({
            'name': 'tuning_threshold_0.3',
            'config': {
                'model_type': 'improved',
                'voxel_resolution': 32,
                'threshold': 0.3,
                'use_features': True,
                'n_points': 10000,
                'f_score_threshold': 0.01,
                'iou_voxel_res': 32
            }
        })

        experiments.append({
            'name': 'tuning_threshold_0.7',
            'config': {
                'model_type': 'improved',
                'voxel_resolution': 32,
                'threshold': 0.7,
                'use_features': True,
                'n_points': 10000,
                'f_score_threshold': 0.01,
                'iou_voxel_res': 32
            }
        })

    #running experiments
    all_results = []
    for exp in experiments:
        result = runner.run_experiment(
            config = exp['config'],
            experiment_name = exp['name'],
            num_samples = num_samples
        )
        if result is not None:
            all_results.append(result)

    #saving comparison
    comparison_file = os.path.join(args.results_dir, 'comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(all_results, f, indent = 2)

    print(f"\nAll experiments completed!")
    print(f"Results saved in: {args.results_dir}/")
    print(f"Comparison saved to: {comparison_file}")

if __name__ == '__main__':
    main()