# Reproducibility

## Frozen environment

The selected project run recorded:

```text
Python  3.10.17
PyTorch 2.10.0+cu128
CUDA    12.8
NumPy   2.2.6
SciPy   1.15.3
GPU     NVIDIA Tesla T4
```

The public package pins the core Python dependencies. GPU results can still vary with driver/CUDA/kernel versions; report the actual environment when reproducing latency.

## Verification levels

### Level 1: source sanity

```bash
pip install -e '.[dev]'
pytest -q
python scripts/smoke_test.py
```

Checks include the 440,562-parameter count, output shapes, mask behavior, corruption-suite size/determinism, and strict checkpoint loading.

### Level 2: frozen checkpoint clean replay

After preparing UTD-MHAD:

```bash
python scripts/evaluate.py --prepared-dir data/processed --device cuda
```

Expected:

```text
correct   414 / 430
accuracy  0.9627906976744186
macro-F1  0.96364139833354
```

### Level 3: frozen checkpoint corruption replay

```bash
python scripts/evaluate_robustness.py \
  --prepared-dir data/processed \
  --device cuda \
  --output outputs/robustness.json
```

Expected:

```text
equal-category accuracy  0.851298449612403
equal-category macro-F1  0.8480595778854244
```

### Level 4: retrain from scratch

```bash
python scripts/train.py --prepared-dir data/processed --output-dir outputs/final --device cuda
```

Training is deterministic-under-seed as far as supported by PyTorch/CUDA, but exact floating-point replay can depend on the environment. Compare final metrics and saved artifacts rather than assuming byte-identical checkpoints across hardware/software stacks.

## Public extraction equivalence audit

Before this folder was prepared for public use, the extracted model was loaded side-by-side with the internal final implementation. The public model accepted the internal model state dict strictly and produced:

```text
internal parameters       440562
public parameters         440562
state-dict keys equal     true
max absolute logit diff   0.0
max availability diff     0.0
predictions equal         true
```

The packaged frozen checkpoint was then evaluated through the **public code itself**, reproducing both the clean and corruption headline numbers above.

## Checkpoint integrity

Exact SHA-256 hashes are recorded in `checkpoints/README.md`.
