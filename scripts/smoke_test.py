#!/usr/bin/env python3
import torch
from safesense import SafeSenseFactorized, parameter_count

model = SafeSenseFactorized().eval()
skeleton = torch.zeros(2, 128, 177)
imu = torch.zeros(2, 128, 12)
mask = torch.ones(2, 128, 2)
with torch.inference_mode():
    logits, weights = model(skeleton, imu, mask)
assert logits.shape == (2, 27)
assert weights.shape == (2, 2)
assert parameter_count(model) == 440_562
print({"logits": list(logits.shape), "weights": list(weights.shape), "parameters": parameter_count(model)})
