# References for the SafeSense live demo

The model research report (`docs/robust_har_research.md`) has 48 more sources, each tied to the claim it supports.
This list covers what the demo code is built on and what the next steps need. Links collected on 28 and 29 October 2026.

## Our own work

- SafeSense paper: `docs/paper/safesense-paper.pdf` (corruption-aware skeleton plus IMU recognition, 25-scenario corruption suite)
- Code repository: https://github.com/Ashok-19/SafeSense (folders `demo/`, `camapp/`)

## Datasets

- NTU RGB+D 120 (Liu et al., TPAMI 2020): https://arxiv.org/abs/1905.04757 , dataset page https://rose1.ntu.edu.sg/dataset/actionRecognition/
- NTU-120 2D HRNet keypoints in PYSKL format (what the demo models are trained and tested on): https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_hrnet.pkl
- PYSKL data format documentation: https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md
- UTD-MHAD (skeleton plus wrist IMU, 27 actions; the paper's dataset, candidate for an IMU demo): https://personal.utdallas.edu/~kehtar/UTD-MHAD.html

## Pose estimation (camera to keypoints)

- rtmlib (what `app.py` uses: YOLOX + RTMPose as ONNX, no mmcv needed): https://github.com/Tau-J/rtmlib
- RTMPose paper: https://arxiv.org/abs/2303.07399
- COCO keypoint format (17 joints): https://cocodataset.org

## Action models

- PYSKL toolbox, ST-GCN++ and its pretrained NTU-120 HRNet weights (Apache-2.0): https://github.com/kennymckormick/pyskl
- PYSKL tech report (ST-GCN++): https://arxiv.org/abs/2205.09443
- ST-GCN++ configs and accuracy table: https://github.com/kennymckormick/pyskl/blob/main/configs/stgcn%2B%2B/README.md
- ST-GCN, the original spatial temporal graph network: https://arxiv.org/abs/1801.07455
- PoseC3D, "Revisiting Skeleton-based Action Recognition" (best published robustness to dropped keypoints on 2D input): https://arxiv.org/abs/2104.13586
- MotionBERT (pretrained motion representations, 2D input): https://arxiv.org/abs/2210.06551

## Robustness, degradation and abstention

- Cormier et al., realistic skeleton augmentation for real-world scenarios, WACVW 2024: https://openaccess.thecvf.com/content/WACV2024W/RWS/papers/Cormier_Enhancing_Skeleton-Based_Action_Recognition_in_Real-World_Scenarios_Through_Realistic_Data_WACVW_2024_paper.pdf
- Skelbumentations (skeleton augmentation library from the same authors): https://github.com/MickaelCormier/Skelbumentations
- SHARDeg, skeleton recognition under frame loss and low frame rate: https://arxiv.org/abs/2505.18048
- RA-GCN, recognition from incomplete skeletons: https://arxiv.org/abs/2008.03791
- Hendrycks and Dietterich, benchmarking robustness to common corruptions: https://arxiv.org/abs/1807.01697
- Selective classification (abstaining with a risk guarantee): https://arxiv.org/abs/1705.08500
- Guo et al., calibration of modern neural networks (temperature scaling): https://proceedings.mlr.press/v70/guo17a
- 1 Euro filter (keypoint smoothing, display only unless tests show a gain): https://cristal.univ-lille.fr/~casiez/1euro

## IMU (next step)

- Android sensors overview (accelerometer, gyroscope, sampling rates): https://developer.android.com/develop/sensors-and-location/sensors/sensors_overview
- Android motion sensors: https://developer.android.com/develop/sensors-and-location/sensors/sensors_motion
- IMUTube, virtual IMU data from video poses (if simulated IMU is ever used, label it on screen): https://arxiv.org/abs/2006.05675

## Libraries the demo uses

- FastAPI: https://fastapi.tiangolo.com
- OpenCV VideoCapture: https://docs.opencv.org
- ONNX Runtime: https://onnxruntime.ai
- PyTorch: https://pytorch.org
- MJPEG over HTTP (multipart/x-mixed-replace): the stream format SafeSense Cam serves, see `camapp/src/com/safesense/cam/MjpegServer.java`
