# Robust real-time skeleton HAR for the SafeSense live demo

Research date: 2026-10-28. Every number below has a source link. "Not found" means it could not be verified.

## Executive summary

1. Among 2023 to 2026 models, almost all state-of-the-art releases (InfoGCN, HD-GCN, Hyperformer, BlockGCN, SkateFormer, ProtoGCN) report NTU-120 on 3D Kinect skeletons. No NTU-120 2D COCO-17 checkpoints were found for them. PYSKL is still the only verified source of NTU-120 HRNet 2D (COCO-17) weights: ST-GCN++, CTR-GCN, MS-G3D, AAGCN, ST-GCN and PoseC3D.
2. The strongest published robustness evidence for 2D input is PoseC3D. When one limb keypoint is dropped in every frame, it loses under 1 point, while MS-G3D loses 14.3 points. PoseC3D has 2.0M parameters and 15.9 GFLOPs, against ST-GCN++ at 1.39M and 2.80 GFLOPs.
3. The PYSKL ST-GCN++ NTU-120 HRNet config uses no noise, joint-drop or frame-drop augmentation. Its 84.4 (joint) comes with a config whose test pipeline averages 10 clips. Live single-window accuracy, and accuracy under degradation, are therefore unverified.
4. Training with realistic skeleton degradations is the best-supported lever. It added 3.0 to 4.6 points for four skeleton models (AGCN, CTR-GCN, PSUMNet) on real-world UAV-Human. MS-G3D with robust training kept 90.6 at full limb-drop, against 77.7 without it.
5. Frame loss and low fps: degradation type matters more than effective frame rate, with gaps of over 40 points at the same rate (SHARDeg). Linear interpolation recovers a lot for models that do not resample to a fixed length.
6. Confidence alone is not a safe abstention signal under domain shift. Skeleton models stay confidently wrong. Gate on input quality as well as on calibrated model confidence.
7. Recommendation: fine-tune PYSKL ST-GCN++ (joint) with the SafeSense corruption suite and make it the main demo model, shown side by side with the stock ST-GCN++ under the existing Degrader. Keep SafeSense-Pose (78.1) as the compact, paper-consistent fallback. Do not switch to a new architecture before the demo.

## 1. Candidate models

Notes on the table. "2D wts" means NTU-120 xsub weights trained on 2D COCO-17 (HRNet) keypoints. PYSKL numbers are top-1 on NTU-120 XSub with HRNet 2D, from the per-model READMEs. Params and FLOPs come from each paper's own setting (often 3D, 25 joints), so compare them only roughly.

| Model | NTU-120 xsub top-1 | Params / FLOPs | Robustness evidence | 2D wts | License |
|---|---|---|---|---|---|
| ST-GCN++ (PYSKL) | 2D: J 84.4, B 84.8, 2s 86.4 [1] | 1.39M / 2.80G [2] | No robustness test found. Config trains without degradation aug [3] | Yes [1] | Apache-2.0 [4] |
| CTR-GCN (PYSKL) | 2D: J 82.2, B 84.6, 2s 85.8 [5] | 1.43M / 2.82G [2] | Corruption aug +4.15 (4s, UAV-Human CSv1) [6] | Yes [5] | PYSKL Apache-2.0 [4]; original repo CC BY-NC 4.0 [7] |
| MS-G3D (PYSKL) | 2D: J 85.5, B 85.0, 2s 86.7 [8] | 3.17M / 10.27G [2] | 92.0 to 77.7 with one limb kp dropped per frame; 90.6 with robust training [9]. Weakest at high frame dropout [10] | Yes [8] | Apache-2.0 [4] |
| AAGCN (PYSKL) | 2D: J 80.2, B 84.2, 2s 84.7 [11] | 3.77M / 6.07G [2] | Corruption aug +4.63 for 4s-AGCN (UAV-Human CSv1) [6] | Yes [11] | Apache-2.0 [4] |
| PoseC3D (PYSKL) | 2D: J 85.9, 2s 86.7 [12] | 2.0M / 15.9G [9] | Under 1 pt drop at full limb-drop (92.4 to 91.5, FineGYM) [9]. Less sensitive to lower-quality poses (paper Table 5) [9] | Yes [12] | Apache-2.0 [4] |
| DG-STGCN (PYSKL) | 3D: J 85.7 [13] | not found | not found | Not found (3D only) [13] | Apache-2.0 [4] |
| InfoGCN / HD-GCN | 3D: J 85.1 / 85.7 [14] | 1.56M, 3.34G / 1.66M, 3.44G [14] | not found | Not found | InfoGCN: not found; HD-GCN MIT [15] |
| Hyperformer | 3D: J 86.6 [14] | 2.71M / 9.64G [14] | not found | Not found | Apache-2.0 [16] |
| SkateFormer (ECCV 2024) | 3D: J 87.7, 2-ens 89.4 [14] | 2.02M / 3.62G, 11.25 ms (GPU) [14] | not found | Not found (3D NTU-120 weights only) [17] | MIT file, but README limits use to research and education [17] |
| BlockGCN (CVPR 2024) | 3D, figure not checked | over 40% fewer GC params [18] | not found | Not found | Apache-2.0 [19] |
| ProtoGCN (CVPR 2025) | 3D: J 85.52, 2-ens 89.75 [20] | not found | not found. Ships Flip and Part Drop aug [20] | Not found (uses ntu120_3danno) [21] | MIT [21] |
| MotionBERT (ICCV 2023) | NTU-120: one-shot only; NTU-60 xsub 93.0 (2D HRNet, x,y,conf) [22] | not stated (DSTformer, depth 5, 512 dim) [22] | Pretrains with 15% joint masking plus noise to mimic occlusion. No corrupted-test numbers [22] | Not found | Apache-2.0 [23] |
| RA-GCN (2019/2020) | 3D NTU-60 CS: 85.9 (3s) [24] | not found | Trunk occluded 50.2 to 70.6; 30 occluded frames 44.5 to 66.3, vs ST-GCN [24] | Not found | not found |
| EfficientGCN-B0 | 3D: 86.6 [25] | 0.29M / 2.73G (NTU-60) [25] | not found | Not found | not found |
| Logsig-RNN family | 3D: 75.8 (stacked) [26] | not found | Most degradation-resistant in SHARDeg; best below 5 fps [10] | Not found | not found |
| Noor et al. CVPRW 2024 | 5 classes only, not NTU-120 [27] | 0.1M params [27] | none [27] | COCO-17 2D, but not NTU-120 | not found |

Other 2024 to 2026 occlusion and corruption work, for reference only: there are no 2D NTU-120 weights for any of these.
- OPSTL: self-supervised learning on occluded NTU-60/120, with KNN imputation of missing joints. Code at cyfml/OPSTL [28].
- Parts-Mamba (2025): claims "up to 12.9%" gain under NTU occlusion settings. Details not verified [29].
- FineTec (2025): temporal-corruption completion on NTU and Gym [30].
- Universal-Skeleton (2026): mixes 2D 17-joint and 3D formats, with no corruption tests [31].
- NTU-X adds richer 118-joint skeletons for subtle actions. It is not a corruption benchmark [32].

Robustness benchmarks worth citing in the demo:
- RA-GCN occlusion protocol: one of five body parts zeroed, or a block of frames zeroed [24].
- SHARDeg: on NTU-120, uniform subsampling to 15/10/5/3 fps, random frame drop of 0 to 90%, and single-block dropout [10].
- Skelbumentations: occlusion, interpolation, swap and mirror perturbations [6][33].
- PoseC3D: limb-keypoint drop with probability p [9].

## 2. Pipeline techniques and evidence

The repo's `demo/app.py` and `demo/robust.py` already have the following. Rows below refer to them.
- A Degrader with missing joints, noise, drop and fps caps.
- Confidence masking at 0.3.
- Gap filling up to 0.6 s and glitch rejection.
- A quality measure and an "input too degraded" gate.
- Nearest-centre tracking.
- An EMA-plus-hysteresis `Smoother`.
- Time-based interpolation to a 30 fps grid.

| Technique | Evidence | Verdict for this demo |
|---|---|---|
| Confidence-aware joint masking | PYSKL's PreNormalize2D zeroes x,y for score <= 0.01 and feeds (x,y,score) [34]. So "missing" is native to the input format, but NTU HRNet training data rarely contains such zeros. ByteTrack shows that low-score detections still carry signal under occlusion [35] | Keep. Pair it with training that contains masked joints (row 6), or the model sees an unfamiliar input |
| Temporal gap filling, interpolation | SHARDeg: interpolation lifts MS-G3D from 13.6% to 56.3% at 5 fps. Models that already resample to a fixed length gain only about 0.8% [10]. Cormier: interpolation as an augmentation added +0.00 to the ensemble [6] | Keep, for short gaps only. Never fill long gaps: show them as missing |
| One Euro or Kalman smoothing of keypoints | One Euro trades jitter against lag with mincutoff and beta [36]. SmoothNet reduces pose jitter but has no action-recognition evaluation [37]. Direct evidence that keypoint smoothing improves skeleton HAR accuracy: not found. NTU HRNet training poses are per-frame. PYSKL's data doc mentions no smoothing [38] | Use for display only, or validate offline first. It can blunt fast motions and shift the input distribution |
| Pose tracking for identity | PYSKL notes that live video needs a tracker and ships a naive similarity tracker [39]. rtmlib has `PoseTracker` with det_frequency [40]. ByteTrack keeps occluded people tracked through low-score boxes [35] | Keep the current tracker. Show the track ID and the number of re-acquisitions on the dashboard |
| Time-based windowing and resampling | ST-GCN++ test pipeline: UniformSample clip_len=100 over the clip [3]. SHARDeg: at the same effective fps, accuracy differs by over 40 points depending on degradation type [10]. Logsig models are reparameterisation-invariant [26] | Keep time-based windows. Report effective fps, not camera fps |
| Training-time degradation aug | +3.0 to +4.6 pts on UAV-Human for 4 skeleton models; +3.59 on JHMDB [6]. MS-G3D robust training: 90.6 vs 77.7 at p=1, costing 1.1 pt clean [9]. Temporal frame-drop aug plus consistency loss raised corrupted mAP (for example 58.33 to 68.29) in action detection [41]. MotionBERT masking plus noise helped accuracy [22] | Highest impact. Mirror the live Degrader exactly: joint and limb drop to (0,0,0), truncation, Gaussian jitter scaled by body size, frame drop, fps resampling, confidence noise |
| Test-time augmentation | PYSKL reports 10-clip test numbers (num_clips=10) [3]. Flip TTA gain for 2D NTU-120: not found | Optional. A few temporal crops of the window are cheap for a 2.8 GFLOP model, but measure latency |
| Abstention and uncertainty | Selective classification sets a confidence threshold for a target risk [42]. Temperature scaling calibrates softmax [43]. Under skeleton domain shift (NTU-120 to Gym2D), the model stays confidently wrong. Energy and Mahalanobis scores detect the shift, but the risk-coverage curve stays poor. A small learned gate helps [44] | Abstain on (input quality below threshold) OR (calibrated max-prob below threshold). Do not rely on softmax alone |
| Prediction smoothing with hysteresis | Common in demos: a mean filter over windows plus a 0.8 threshold [45]. Controlled HAR studies: not found | Keep. Its cost is latency, so show "label changed after N windows" |

## 3. How robustness is presented

- Severity curves and relative robustness. ImageNet-C style work reports accuracy per corruption and severity, and "relative corruption robustness" [46]. THUMOS14-C reports a relative robustness score per model (54.16% to 84.31%) [41]. A dashboard can show the same number: accuracy under the current corruption divided by clean accuracy, from offline NTU-120 replay.
- Side-by-side clips at increasing severity, and the same model on clean and corrupted input. 3D Common Corruptions shows intensity videos and model-degradation videos. Its live demo runs on clean uploads only, not on injected corruptions [47].
- Honest limits. One widely forked real-time skeleton HAR repo warns that it is only good "for course demo". It fills missing joints from the previous frame and labels only above a smoothed 0.8 score [45]. PYSKL's own NTU-120 skeleton demo is offline GPU, and its only real-time CPU demo is hand gestures [39].
- Live corruption injection with a robust vs baseline split screen. No published skeleton-HAR system doing this was found. The SafeSense demo already has a Degrader and compare mode, so the novel part is showing both live and pairing them with offline curves.

## 4. Recommendation

Model. The main live model should be PYSKL ST-GCN++ joint, fine-tuned with the SafeSense corruption suite and started from the released checkpoint [1][3]. Show it next to the stock ST-GCN++ checkpoint as the baseline.

Why:
- It has the best verified 2D NTU-120 accuracy that fits CPU real time (2.80 GFLOPs [2]).
- The stock checkpoint saw no degradations in training [3]. The literature shows the largest, best-replicated robustness gains come from degradation-matched training [6][9].
- The baseline vs fine-tuned comparison isolates exactly the paper's claim, with the same architecture and the same pose stream.
- If time runs out, run stock ST-GCN++ (accuracy) and corruption-aware SafeSense-Pose (78.1) side by side. Let measured offline curves, not expectation, decide which one the demo calls "robust".
- PoseC3D is the evidence-backed robust alternative [9], but it costs about 5.7 times the FLOPs of ST-GCN++ [2][9] and needs heatmap rendering. Use it only as a stretch goal.
- None of the 2024 to 2026 SOTA models offer 2D COCO-17 NTU-120 weights.

Top 5 pipeline changes, ranked by expected impact on a convincing, honest demo:
1. **Corruption-aware fine-tune of ST-GCN++ (joint, then bone).**
   - Use the same transforms as `robust.Degrader`, plus truncation (person partly out of view) and confidence jitter.
   - Encode a missing joint as (0,0,0), as PreNormalize2D does [34].
   - Evidence: [6][9][22]. Expect a small cost on clean accuracy (MS-G3D lost 1.1 points [9]); report it.
2. **Offline robustness curves on NTU-120 xsub val, through the exact live preprocessing.**
   - Cover each Degrader condition and severity, single window, no 10-clip TTA.
   - Do this for stock ST-GCN++, fine-tuned ST-GCN++, SafeSense-Pose and SafeSense-orig.
   - Show the curve on the dashboard with a marker at the live condition, plus relative robustness [41][46]. This is what makes the live demo honest, because the live NTU-domain shift is otherwise unmeasured [44].
3. **Quality-gated abstention.**
   - Temperature-scale each model on NTU val [43].
   - Abstain when the quality index (visible-joint fraction, mean score, effective fps, longest gap, track stability) or the calibrated confidence is below a threshold chosen from a risk-coverage curve [42][44].
   - Show "uncertain: legs not visible" rather than a wrong label.
4. **Time-aware gap handling.**
   - Interpolate gaps up to 0.6 s by timestamp; mark longer gaps as missing. Never fabricate them.
   - Show effective fps and the share of interpolated frames, since SHARDeg shows the drop pattern matters [10].
   - Keep keypoint smoothing out of the model path unless offline curves show a gain [36][37].
5. **Split-screen live comparison with scripted injections.**
   - Same pose stream, baseline vs robust model, same Degrader preset.
   - The presenter walks through clean, limb hidden, 50% frame loss, 5 fps, and composite.
   - The dashboard shows each model's label, confidence, abstentions and the offline expected accuracy for that preset.
   - Also include one real (not injected) occlusion and one walk partly out of frame, to show the injected faults match real ones.

Caveats to state on stage:
- NTU-120 is lab data, so live phone-camera accuracy will be lower than the reported 84.4, which also includes 10-clip testing [3].
- Confidence under domain shift is unreliable [44].
- Injected faults are approximations of real pose-estimator failures [6].

## References

1. PYSKL ST-GCN++ README: https://github.com/kennymckormick/pyskl/blob/main/configs/stgcn%2B%2B/README.md
2. PYSKL tech report (Tables 1 and 2): https://arxiv.org/pdf/2205.09443
3. ST-GCN++ NTU-120 HRNet joint config: https://github.com/kennymckormick/pyskl/blob/main/configs/stgcn%2B%2B/stgcn%2B%2B_ntu120_xsub_hrnet/j.py
4. PYSKL repo (Apache-2.0, model list): https://github.com/kennymckormick/pyskl
5. PYSKL CTR-GCN README: https://github.com/kennymckormick/pyskl/blob/main/configs/ctrgcn/README.md
6. Cormier et al., WACVW 2024: https://openaccess.thecvf.com/content/WACV2024W/RWS/papers/Cormier_Enhancing_Skeleton-Based_Action_Recognition_in_Real-World_Scenarios_Through_Realistic_Data_WACVW_2024_paper.pdf
7. CTR-GCN license: https://github.com/Uason-Chen/CTR-GCN/blob/main/LICENSE
8. PYSKL MS-G3D README: https://github.com/kennymckormick/pyskl/blob/main/configs/msg3d/README.md
9. PoseC3D paper (Tables 3, 4, 5, 8): https://arxiv.org/pdf/2104.13586v2
10. SHARDeg: https://www.arxiv.org/pdf/2505.18048v2
11. PYSKL AAGCN README: https://github.com/kennymckormick/pyskl/blob/main/configs/aagcn/README.md
12. PYSKL PoseC3D README: https://github.com/kennymckormick/pyskl/blob/main/configs/posec3d/README.md
13. PYSKL DG-STGCN README: https://github.com/kennymckormick/pyskl/blob/main/configs/dgstgcn/README.md
14. SkateFormer paper (Tables 1 to 3, 10): https://arxiv.org/pdf/2403.09508v1
15. HD-GCN license: https://github.com/Jho-Yonsei/HD-GCN/blob/main/LICENSE
16. Hyperformer license: https://github.com/ZhouYuxuanYX/Hyperformer/blob/main/LICENSE
17. SkateFormer repo: https://github.com/KAIST-VICLab/SkateFormer
18. BlockGCN paper: https://arxiv.org/abs/2305.11468v3
19. BlockGCN repo: https://github.com/ZhouYuxuanYX/BlockGCN
20. ProtoGCN model doc: https://github.com/firework8/ProtoGCN/blob/main/data/README.md
21. ProtoGCN repo: https://github.com/firework8/ProtoGCN
22. MotionBERT paper: https://arxiv.org/pdf/2210.06551v3
23. MotionBERT license: https://github.com/Walter0807/MotionBERT/blob/main/LICENSE
24. RA-GCN (incomplete skeletons): https://ar5iv.labs.arxiv.org/html/1905.06774 and https://arxiv.org/abs/2008.03791
25. EfficientGCN: https://arxiv.org/pdf/2106.15125
26. Logsig-RNN, BMVC 2021: https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/0724.pdf
27. Noor et al., CVPRW 2024: https://openaccess.thecvf.com/content/CVPR2024W/MAI/papers/Noor_Efficient_Skeleton-Based_Action_Recognition_for_Real-Time_Embedded_Systems_CVPRW_2024_paper.pdf
28. OPSTL: https://arxiv.org/abs/2309.12029
29. Parts-Mamba: https://www.alphaxiv.org/abs/2511.16860
30. FineTec: https://arxiv.org/abs/2512.25067
31. Towards Universal Skeleton-Based Action Recognition: https://arxiv.org/html/2604.17013v1
32. NTU-X: https://arxiv.org/abs/2101.11529
33. Skelbumentations: https://github.com/MickaelCormier/Skelbumentations
34. PYSKL PreNormalize2D and GenSkeFeat: https://github.com/kennymckormick/pyskl/blob/main/pyskl/datasets/pipelines/pose_related.py
35. ByteTrack: https://arxiv.org/abs/2110.06864v2
36. 1 Euro filter, CHI 2012: https://cristal.univ-lille.fr/~casiez/1euro
37. SmoothNet: https://arxiv.org/abs/2112.13715
38. PYSKL data doc: https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md
39. PYSKL demo doc: https://github.com/kennymckormick/pyskl/blob/main/demo/demo.md
40. rtmlib: https://github.com/Tau-J/rtmlib
41. Temporal action detection robustness benchmark, CVPR 2024: https://arxiv.org/pdf/2403.20254
42. Selective classification for DNNs: https://arxiv.org/abs/1705.08500
43. On calibration of modern neural networks: https://proceedings.mlr.press/v70/guo17a
44. Uncertainty failure under skeleton domain shift: https://arxiv.org/abs/2603.15574
45. felixchenfy Realtime-Action-Recognition: https://github.com/felixchenfy/Realtime-Action-Recognition
46. Hendrycks and Dietterich, common corruptions: https://arxiv.org/abs/1807.01697v4
47. 3D Common Corruptions: https://3dcommoncorruptions.epfl.ch/
48. RTMPose (75.8 AP, 90+ FPS on i7-11700 CPU), for pose-side context: https://arxiv.org/abs/2303.07399v2
