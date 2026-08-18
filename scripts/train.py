#!/usr/bin/env python3
from argparse import ArgumentParser
import json
from safesense.training import TrainingConfig, train_final, train_odd_fold

p = ArgumentParser(description="Train the final SafeSenseFactorized configuration")
p.add_argument("--prepared-dir", default="data/processed")
p.add_argument("--output-dir", default="outputs/train")
p.add_argument("--device", default="cuda")
p.add_argument("--fold-subject", type=int, choices=[1, 3, 5, 7])
p.add_argument("--seed", type=int, default=22)
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--learning-rate", type=float, default=1e-3)
p.add_argument("--weight-decay", type=float, default=1e-4)
p.add_argument("--swa-start", type=int, default=45)
a = p.parse_args()
config = TrainingConfig(seed=a.seed, epochs=a.epochs, batch_size=a.batch_size, learning_rate=a.learning_rate, weight_decay=a.weight_decay, swa_start_epoch=a.swa_start)
if a.fold_subject is None:
    result = train_final(a.prepared_dir, a.output_dir, config=config, device=a.device)
else:
    result = train_odd_fold(a.prepared_dir, a.output_dir, validation_subject=a.fold_subject, config=config, device=a.device)
print(json.dumps(result, indent=2))
