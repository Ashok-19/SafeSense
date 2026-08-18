# Pretrained checkpoint

This directory contains the **frozen SafeSenseFactorized seed-22 late-SWA model** used for the reported UTD-MHAD results.

| File | Purpose | Bytes | SHA-256 |
|---|---|---:|---|
| `safesense_seed22_swa.pt` | PyTorch state dict | 1814707 | `4675446cd29b3670d24e0f018f8a8ee2c4f3fd7e4b5d7296b15bdbf8f5de2c57` |
| `safesense_seed22_stats.npz` | train-only normalization/noise statistics | 3084 | `21f035dfe0e0da47c4831f705eb395fb56760329eb8bde7b552e8f594a548084` |

Training configuration: 60 epochs, AdamW, seed 22, equal weight averaging over completed epochs 45-60 inclusive.

The checkpoint has been replayed through the code in this public folder and reproduces **414/430 = 96.2791% clean accuracy**, **96.3641% macro-F1**, and **85.1298% equal-category corruption accuracy** on the fixed 25-scenario suite.
