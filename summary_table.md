# 3D Reconstruction Evaluation Results

## Summary Table

| Experiment | Chamfer ↓ | F-Score ↑ | IoU ↑ | Successful |
|------------|-----------|-----------|-------|------------|
| baseline_simple | 0.0949±0.0321 | 0.0043±0.0047 | 0.0148±0.0092 | 100/3839 |
| baseline_improved | 0.1026±0.0355 | 0.0040±0.0045 | 0.0152±0.0098 | 100/3839 |

## Key Findings

- **Best Chamfer Distance**: baseline_simple (0.094869)
- **Best F-Score**: baseline_simple (0.004297)
- **Best IoU**: baseline_improved (0.015217)

## Hyperparameter Analysis

### Voxel Resolution
- Increasing voxel resolution from 32 to 64 improved F-Score
- Trade-off: Higher resolution increases computation time

### Threshold Value
- Threshold affects the density of reconstructed voxels
- Optimal threshold depends on the target metric
