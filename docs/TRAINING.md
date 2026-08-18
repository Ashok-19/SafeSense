# Training

## Selected configuration

| Setting | Value |
|---|---:|
| Model | `SafeSenseFactorized` |
| Seed | 22 |
| Epochs | 60 |
| Batch size | 32 |
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Gradient clip | 1.0 |
| SWA window | epochs 45-60 inclusive |

The final run fits normalization and physical-space noise statistics on subjects 1, 3, 5, and 7, trains for all 60 epochs, averages the model states from epochs 45 through 60, and evaluates the resulting single SWA checkpoint.

## Four training views

Every training batch is evaluated as four label-preserving views:

- clean;
- temporal corruption;
- structured physical-space corruption;
- subject/sensor-style perturbation.

Main classification loss:

```text
L_cls = 0.55 CE(clean)
      + 0.15 CE(temporal)
      + 0.15 CE(structured)
      + 0.15 CE(style)
```

Full objective:

```text
L = L_cls
  + 0.15 L_expert
  + 0.15 L_alignment
  + 0.10 L_consistency
```

### Expert supervision

Skeleton-only and IMU-only auxiliary heads are trained to predict the same activity. This encourages both modalities to remain individually discriminative.

### Cross-modal alignment

When both sensors are available, mean skeleton and IMU embeddings are normalized and encouraged to have high cosine similarity.

### Consistency

The clean prediction distribution acts as a detached teacher for temporal, structured, and style-perturbed views using temperature-scaled KL divergence.

## Temporal/structured corruption families

Training samples can receive:

- relative shifts of ±1..±4 frames;
- 10%, 30%, or 50% frame loss;
- Gaussian noise at standardized strengths 0.1, 0.3, or 0.5;
- complete modality dropout;
- composite IMU shift + 30% frame loss + 0.3 noise.

The structured path perturbs raw physical channels, then rebuilds position/motion/bone and IMU-difference features and renormalizes them.

## Subject/sensor style perturbation

The style view applies label-preserving changes in physical feature space:

- body scale 0.85-1.15;
- yaw perturbation ±15 degrees;
- small translation noise;
- matching rotation of IMU triplets;
- IMU gain 0.90-1.10;
- IMU bias noise.

## Late SWA

The final checkpoint is:

```text
theta_SWA = mean(theta_45, ..., theta_60)
```

It is **one averaged parameter state**, not a 16-model inference ensemble.

## Commands

Final training:

```bash
python scripts/train.py --prepared-dir data/processed --output-dir outputs/final --device cuda
```

One development fold:

```bash
python scripts/train.py --prepared-dir data/processed --output-dir outputs/fold7 --fold-subject 7 --device cuda
```

For paper-scale development, repeat the four odd folds and desired seeds explicitly. Never choose hyperparameters from the even-subject result.
