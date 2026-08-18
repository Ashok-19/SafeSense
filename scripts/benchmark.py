#!/usr/bin/env python3
from argparse import ArgumentParser
import json
import numpy as np
import torch
from safesense.corruptions import base_temporal_masks
from safesense.data import feature_arrays, load_stats
from safesense.efficiency import CudaGraphBatch1Inference, benchmark_model, checkpoint_bytes
from safesense.evaluation import load_model

p = ArgumentParser(description="Batch-1 model-only SafeSense latency benchmark")
p.add_argument("--prepared-dir", default="data/processed")
p.add_argument("--checkpoint", default="checkpoints/safesense_seed22_swa.pt")
p.add_argument("--stats", default="checkpoints/safesense_seed22_stats.npz")
p.add_argument("--device", default="cuda")
p.add_argument("--cuda-graph", action="store_true")
p.add_argument("--warmups", type=int, default=200)
p.add_argument("--iterations", type=int, default=1000)
a = p.parse_args()

fused = np.load(f"{a.prepared_dir}/skeleton_val_features.npy")[:1]
masks = base_temporal_masks(fused)
pair = np.stack((masks["skeleton"], masks["imu"]), axis=-1).astype(np.float32)
skeleton, imu, _ = feature_arrays(fused)
stats = load_stats(a.stats)
skeleton = ((skeleton - stats["skeleton_mean"]) / stats["skeleton_std"]).astype(np.float32) * pair[..., 0, None]
imu = ((imu - stats["imu_mean"]) / stats["imu_std"]).astype(np.float32) * pair[..., 1, None]
model, device = load_model(a.checkpoint, device=a.device)
inputs = tuple(torch.from_numpy(x).to(device) for x in (skeleton, imu, pair))
if a.cuda_graph:
    if device.type != "cuda":
        raise RuntimeError("--cuda-graph requires CUDA")
    model = CudaGraphBatch1Inference(model, inputs)
result = benchmark_model(model, inputs, device=str(device), warmups=a.warmups, iterations=a.iterations, checkpoint_bytes=checkpoint_bytes(a.checkpoint))
print(json.dumps(result, indent=2))
