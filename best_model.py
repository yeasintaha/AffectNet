import os
import json

runs_dir = "runs"
rows = []

for name in os.listdir(runs_dir):
    run_path = os.path.join(runs_dir, name)
    if not os.path.isdir(run_path):
        continue
    results_file = os.path.join(run_path, "test_results.json")
    if not os.path.exists(results_file):
        continue
    with open(results_file) as f:
        d = json.load(f)
    rows.append((d.get("macro_f1", 0), d.get("accuracy", 0), name))

rows.sort(reverse=True) 

print(f"{'macro-F1':>9} {'accuracy':>9}  run")
print("-" * 60)
for mf1, acc, name in rows:
    print(f"{mf1:9.4f} {acc:9.4f}  {name}")

if rows:
    best = rows[0]
    print(f"\nBest model: {best[2]}")
    print(f"LOcation: runs/{best[2]}/best.pt")
