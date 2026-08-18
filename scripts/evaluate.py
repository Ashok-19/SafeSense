#!/usr/bin/env python3
from argparse import ArgumentParser
import json
from safesense.evaluation import evaluate_clean

p = ArgumentParser(description="Evaluate a frozen SafeSense checkpoint on the even-subject clean split")
p.add_argument("--prepared-dir", default="data/processed")
p.add_argument("--checkpoint", default="checkpoints/safesense_seed22_swa.pt")
p.add_argument("--stats", default="checkpoints/safesense_seed22_stats.npz")
p.add_argument("--device", default="cuda")
a = p.parse_args()
print(json.dumps(evaluate_clean(a.prepared_dir, a.checkpoint, a.stats, device=a.device), indent=2))
