#!/usr/bin/env python3
from argparse import ArgumentParser
import json
from safesense.evaluation import evaluate_robustness, write_json

p = ArgumentParser(description="Evaluate the fixed 25-scenario SafeSense corruption suite")
p.add_argument("--prepared-dir", default="data/processed")
p.add_argument("--checkpoint", default="checkpoints/safesense_seed22_swa.pt")
p.add_argument("--stats", default="checkpoints/safesense_seed22_stats.npz")
p.add_argument("--device", default="cuda")
p.add_argument("--seed", type=int, default=20260814)
p.add_argument("--output", default="outputs/robustness.json")
a = p.parse_args()
result = evaluate_robustness(a.prepared_dir, a.checkpoint, a.stats, device=a.device, corruption_seed=a.seed)
write_json(a.output, result)
print(json.dumps({"equal_category_accuracy": result["equal_category_corrupted_accuracy"], "equal_category_macro_f1": result["equal_category_corrupted_macro_f1"]}, indent=2))
