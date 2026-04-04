import json

with open('results/comparison.json', 'r') as f:
    results = json.load(f)

print("=" * 70)
print("FULL BASELINE RESULTS (3,839 samples)")
print("=" * 70)

for exp in results:
    print(f"\n{exp['experiment_name']}:")
    print(f"  Chamfer Distance: {exp['metrics']['chamfer_distance']['mean']:.6f}")
    print(f"  F-Score:          {exp['metrics']['f_score']['mean']:.6f}")
    print(f"  IoU:              {exp['metrics']['iou']['mean']:.6f}")