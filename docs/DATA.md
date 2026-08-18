# Data

## Dataset

SafeSense uses UTD-MHAD, introduced by Chen, Jafari, and Kehtarnavaz (ICIP 2015, DOI `10.1109/ICIP.2015.7350781`). The released dataset contains RGB, depth, skeleton, and inertial modalities; SafeSense reads **only skeleton and inertial files**.

Repository-verified usable skeleton/IMU pairs:

| Item | Value |
|---|---:|
| Action classes | 27 |
| Subjects | 8 |
| Nominal trials/action/subject | 4 |
| Paired usable samples | 861 |
| Odd-subject samples | 431 |
| Even-subject samples | 430 |

The three absent/corrupt nominal trials are `a8_s1_t4`, `a23_s6_t4`, and `a27_s8_t4`, which is why the released aligned count is 861 rather than 864.

## Cross-subject protocol

```text
training/development: subjects 1, 3, 5, 7
final evaluation:     subjects 2, 4, 6, 8
```

Within the odd-subject side, the development helper supports leave-one-odd-subject-out folds:

```text
validate 1 -> train 3,5,7
validate 3 -> train 1,5,7
validate 5 -> train 1,3,7
validate 7 -> train 1,3,5
```

Normalization/noise statistics are fitted from the relevant training subjects only.

## Raw file layout

Do not commit the UTD-MHAD archives or extracted recordings. A typical local layout is:

```text
data/raw/
├── Skeleton/
│   ├── a1_s1_t1_skeleton.mat
│   └── ...
└── Inertial/
    ├── a1_s1_t1_inertial.mat
    └── ...
```

Run:

```bash
python scripts/prepare_utd_mhad.py --data-dir data/raw --output-dir data/processed
```

## Temporal preparation

For each paired sample:

1. skeleton data are read as 20 XYZ joints;
2. sequences are padded/repeated to 128 frames following the audited 128-frame preparation behavior used by this release;
3. skeleton translation/orientation is normalized;
4. the six-channel IMU stream is nearest-index aligned to the original skeleton timeline;
5. the IMU channels are stored as two 3-D pseudo-joints in the prepared tensor.

Prepared shape:

```text
[N, 1, 128, 22, 3]
                 ├── joints 0..19: skeleton
                 └── joints 20..21: 6 IMU values
```

These pseudo-joints are only a preparation container. The SafeSense model immediately separates the modalities and does **not** treat the IMU as anatomical joints.

## SafeSense feature construction

### Skeleton: 177 dimensions/frame

```text
20 XYZ joint positions       60
20 XYZ first differences     60
19 XYZ bone vectors          57
--------------------------------
total                       177
```

### IMU: 12 dimensions/frame

```text
prepared inertial channels    6
first differences             6
--------------------------------
total                        12
```

The implementation is in `src/safesense/data.py::feature_arrays`.

## Masks

The prepared IMU region provides the clean temporal validity signal. During corruption evaluation, skeleton and IMU receive separate masks so a missing or shifted sensor cannot contribute as if it were valid.
