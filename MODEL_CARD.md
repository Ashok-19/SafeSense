# SafeSense model card

## Model

`SafeSenseFactorized` is a 440,562-parameter sequence classifier for isolated human activity recognition. It consumes synchronized 3D skeleton and IMU sequences and predicts one of 27 UTD-MHAD action classes.

## Intended use

Research on multimodal human activity recognition, sensor-degradation robustness, skeleton/IMU fusion, and compact inference.

## Inputs

- 20-joint 3D skeleton trajectories prepared to 128 time steps;
- six inertial channels aligned to the skeleton timeline;
- validity masks.

The released helper converts these into 177 skeleton and 12 IMU features per time step.

## Outputs

- 27 class logits;
- a two-element modality-availability weight vector.

The availability vector is not calibrated class confidence.

## Training

UTD-MHAD odd subjects `1,3,5,7`, seed 22, 60 epochs, AdamW, four-view corruption/style training, auxiliary expert/alignment/consistency losses, and equal SWA over epochs 45-60.

## Evaluation

Even subjects `2,4,6,8`, 430 samples:

- 96.2791% clean accuracy;
- 96.3641% macro-F1;
- 85.1298% equal-category corruption accuracy over the fixed 25-scenario suite.

## Limitations

- UTD-MHAD is small: 8 subjects and 861 usable paired samples.
- The task is isolated sequence classification, not continuous streaming action detection.
- Corruptions are controlled synthetic perturbations, not recordings of naturally failing hardware.
- The model uses skeleton+IMU only; RGB/depth systems can exploit cues unavailable here.
- The 1.732 ms T4 result is model-only and excludes sensing, skeleton estimation, preprocessing, and host-to-device transfer.
- RGB-free sensing is not a formal privacy guarantee; skeleton trajectories can still contain sensitive information.
- The model should not be used as a safety-critical decision system without domain-specific validation.

## Comparison boundary

SafeSense is not an overall UTD-MHAD accuracy-SOTA claim. Its primary contribution is the joint clean/robustness/efficiency operating point under the stated sensor contract.
