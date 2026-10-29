# SafeSense live demo: handover guide

Read this first. It explains what the demo is for, how it works, how to run it, where everything is, what is known to
be weak, and what to improve next. Everything needed is in this folder, except the NTU-120 skeleton file (a large
download, link in section 4).

Contents of this pack:

| Folder | What it is |
|---|---|
| `demo/` | The live demo: Python server, browser dashboard, robust pipeline, tests, and the trained SafeSense-Pose model |
| `camapp/` | SafeSense Cam, our own minimal Android app that streams the phone camera to the laptop (source, build script, APK, signing key) |
| `training/` | Kaggle notebooks that trained the demo models on NTU-120 |
| `docs/` | The SafeSense paper (the reason for all this), model research report, per-action accuracy table, screenshots |
| `REFERENCES.md` | Online references: datasets, models, libraries, papers, Android docs |

---

## 1. What the demo must show (read this before changing anything)

SafeSense is our final-year project: **robust, efficient human activity recognition when the sensor data is
degraded**. The paper (`docs/paper/safesense-paper.pdf`) trains a compact skeleton plus IMU model with
corruption-aware training and evaluates it on a 25-scenario corruption suite: temporal shift, frame loss (10, 30, 50
percent), sensor noise, missing modality, and composite faults.

So the live demo is **not** just "the model guesses the action". Its job is to show, in real time, that the pipeline
plus the model keeps working when the input gets worse, and says so honestly when the input is too bad. Every change
should serve that story:

1. The audience sees the person, the skeleton and the recognised action.
2. The presenter degrades the input on purpose (hide an arm, add noise, lose frames, drop the frame rate).
3. The audience sees the input quality drop, and sees SafeSense keep (or honestly lose) the right answer, next to a
   comparison line on exactly the same input.
4. The claims are backed by numbers measured offline on real NTU-120 test clips through the same code
   (Tests > Robustness test), shown on the same screen.

Project rules for anything shown or written: never invent numbers, latency or citations; label what is measured and
what is not; no "state of the art" or "first ever" claims.

---

## 2. How it works

```
 Phone (SafeSense Cam, MJPEG over Wi-Fi, port 8080)
   |
   v
 camera_loop        OpenCV reads http://<phone>:8080/video, reconnects on failure
   |
 pose_loop          RTMPose via rtmlib (onnxruntime, CPU): 17 COCO keypoints + scores per person, about 10 fps on a laptop
   |                select_people: keep the 2 biggest confidently detected people
   |                Degrader (robust.py): the injected faults, only when the presenter turns one on
   v
 buf                the last `window` seconds (default 3 s) of (time, keypoints, scores)
   |
 action_loop        every 0.25 s:
   |                  robust.measure_quality   input quality: joints visible, pose confidence, fps, steadiness
   |                  track                    keep each person in the same slot across frames
   |                  robust.clean_window      the robust pipeline (on/off switch): low-confidence joints become
   |                                           missing, glitches rejected, gaps up to 0.6 s filled in time
   |                  resample to 48 steps     evenly in time (training samples 48 segment centres per clip)
   |                  pack / normalise         centre on the hips, scale by the torso (same code as training)
   |                  model                    SafeSense-Pose (default), original SafeSense, or PYSKL ST-GCN++
   |                  comparison line          same model without the robust pipeline (default), or another model
   |                  Smoother                 moving average, confidence threshold, label changes only after 2 wins
   |                  abstain                  "input too degraded" when under 35% of joints are visible or fps < 2
   v
 FastAPI (app.py)   /state, /stream.mjpg, /api/... endpoints
   |
   v
 dashboard.html     Demo page, Setup, Record and check, Tests (plain HTML + JavaScript, no build step)
```

### Files in `demo/`

| File | Role |
|---|---|
| `app.py` | The server: camera, pose, action loops, settings, all HTTP endpoints, background test jobs |
| `dashboard.html` | The whole user interface, polls `/state` four times a second |
| `robust.py` | `measure_quality`, `Degrader` and `CONDITIONS` (the stress conditions), `clean_window` (the robust pipeline) |
| `safesense_pose.py` | SafeSense-Pose model (ST-GCN style, COCO-17, 2 bodies, 48 steps, inputs x, y, score, dx, dy) and the shared preprocessing (`pack`, `normalise`, `sample_frames`) used by training and live alike |
| `safesense_orig.py` | The original SafeSense architecture adapted to COCO-17 (`SafeSenseCOCO`); it has an IMU branch that is masked off today |
| `pyskl_stgcnpp.py` | Loader for the pretrained PYSKL ST-GCN++ weights (`get_stgcnpp.sh` downloads them) |
| `safesense_pose_ntu120.pt` | Trained SafeSense-Pose checkpoint (682,408 parameters) |
| `robustness_test.py` | Offline robustness test: NTU-120 test clips x stress conditions x pipeline on/off x models; writes `recordings/robustness.json`, which the Demo page shows |
| `replay_test.py` | Replays NTU test clips through the live code at a chosen fps, compares with the offline evaluation |
| `ablate.py` | Step-by-step check of each live preprocessing step on whole clips |
| `prepare_ntu.py` | Builds `ntu_reference.npz` (how NTU people perform each action) for Record and check |
| `analysis.py` | Scores guided recordings against the NTU reference, with hints ("you moved much less than the NTU people") |
| `record_actions.py` | Older terminal version of guided recording (the dashboard does this now) |
| `start.sh` / `setup.sh` | Start the dashboard (creates `.venv` on first run); `start.sh --install` adds a menu entry |
| `run.sh` | Older start script with the camera address on the command line |
| `README.md` | Laptop and phone setup in more detail |

### Main HTTP endpoints (`app.py`)

`/state` (everything the dashboard shows), `/stream.mjpg` (annotated video), `/api/view` (the normalised window the
model sees), `/api/settings` GET/POST, `/api/camera_test`, `/api/find_camera` (scans the laptop's /24 networks for
port 8080), `/api/degrade` (set a stress condition), `/api/stress` (automatic stress test), `/api/robustness`
(offline result), `/api/record`, `/api/recordings`, `/api/recording_view`, `/api/rescore`, `/api/jobs/run|stop`,
`/api/jobs`, `/api/ping`. Settings persist in `settings.json` next to `app.py`.

---

## 3. Models

| Model | File | Params | NTU-120 cross-subject top-1 | Notes |
|---|---|---|---|---|
| SafeSense-Pose | `demo/safesense_pose_ntu120.pt` | 682,408 | 78.10% (top-5 95.29%) | Default. Trained by `training/04_train_safesense_pose_ntu120.ipynb`, 100 epochs on a Kaggle T4 |
| Original SafeSense (COCO) | `demo/safesense_orig_ntu120.pt` (output of notebook 06) | 495,753 | 73.98% (top-5 94.01%) | The paper's architecture, IMU branch masked. Better on hand waving, drinking, clapping; weaker on two-person actions |
| PYSKL ST-GCN++ | `stgcnpp_j.pth`, `stgcnpp_b.pth` via `bash demo/get_stgcnpp.sh` | about 1.4M | 84.4 joint, 84.8 bone, 86.4 both (PYSKL's own numbers, 10-clip test) | Pretrained, Apache-2.0. Not trained with any corruption |

`training/05_train_safesense_pose_ntu120_449k.ipynb` is a smaller SafeSense-Pose variant that has not been run.
Per-action accuracy of the first two models: `docs/action_accuracy.md`. The demo restricts the choice to 37
single-person actions that SafeSense-Pose recognises well (`STRONG` in `app.py`); Setup can switch to all 120.

---

## 4. Run it

Laptop (Linux, Python 3.10 or newer, CPU is enough):

```bash
cd demo
bash start.sh            # first run creates .venv and installs requirements.txt, then opens http://localhost:8765
bash start.sh --install  # optional: adds "SafeSense Demo" to the applications menu
python app.py --mock     # no camera: a synthetic skeleton, to check the install
```

The first real run downloads the RTMPose ONNX models once (rtmlib does this). Port 8765 can be changed with
`SAFESENSE_PORT=9000 bash start.sh`.

Phone: install `camapp/release/SafeSenseCam.apk` (allow installing from that source). Open it; it starts serving at
once and shows its address. Put the laptop on the same Wi-Fi or on the phone's hotspot (college Wi-Fi often blocks
devices from seeing each other; the hotspot always works). The Demo page finds the phone by itself; press **Use**.

Demo flow: stand 2 to 3 m from the camera, whole body in view, do an action, then use the **Stress the input**
buttons or **Run automatic stress test** (keep doing one action for about 40 s). Screenshots: `docs/screenshots/`.

NTU-120 data for the tests (2D HRNet skeletons, PYSKL format):
`https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_hrnet.pkl`, saved as `demo/data/ntu120_hrnet.pkl`.
Then in the dashboard: Tests > **Robustness test** (about 600 clips, a few minutes on CPU) and
**Prepare NTU reference** (needed by Record and check).

Android app build (no Android Studio needed): see `camapp/README.md`. `camapp/release.keystore` (password
`safesense`, alias `safesense`, kept out of the public repository) must be used for every update, or phones refuse to
upgrade the installed app. If it is missing from your copy, ask Ashok for it.

---

## 5. What is known, and what is not yet known

Measured so far:

- For one person, the live input matches the offline evaluation input (replay test on real NTU clips).
- At about 10 fps, interpolating poses in time cost about 7.5 points against 30 fps in earlier replay runs; adding
  keypoint jitter recovered it, which suggests the model expects slightly noisy input. Person handling cost about 1.2
  points.
- In Ashok's own recordings, the movements were much smaller than NTU people's (nod head: movement 0.09 against 1.49 for NTU;
  hand waving 0.6 against 1.8). Small, careful movements are recognised far worse than NTU-style ones. Actions that
  worked: salute, cross arms, arm circles (partly).

**Not yet measured, and the first thing to do:** whether the robust pipeline actually helps. `robustness_test.py`
has only been run on dummy data. Run it on the real NTU file. Check two things: (a) under the stress conditions,
"with pipeline" should beat "without"; (b) on clean input it must not be noticeably worse. If (b) fails, the
confidence cut-off (`CONF_MIN = 0.3` in `robust.py`) is the first suspect, because NTU HRNet scores were never cut
during training.

---

## 6. What to fix and improve (ranked)

1. **Validate the robust pipeline** (section 5) and tune `CONF_MIN`, `GAP_FILL_S`, `GLITCH_TORSO` in `robust.py` on
   the robustness test, not by eye.
2. **Corruption-aware fine-tuning**, the paper's core idea, applied to the demo model. The research report recommends
   fine-tuning PYSKL ST-GCN++ (joint stream) from its released checkpoint with the same faults the Degrader injects
   (hidden limbs, noise, frame loss, low fps, people partly out of frame, confidence jitter), missing joints encoded
   as (0, 0, 0). Then show it next to stock ST-GCN++ (Setup > comparison line) under the same faults. SafeSense-Pose
   can be fine-tuned the same way with notebook 04 (add the degradations to its training augmentation).
3. **Abstention**: today it fires only on input quality (visible joints, fps). Add temperature-scaled model
   confidence and pick the threshold from a risk-coverage curve on NTU test clips.
4. **Quality score calibration**: the weights and Good/Fair/Poor thresholds in `measure_quality` are hand-set. Hiding
   one arm's elbow and wrist still reads "Good" (about 89). Calibrate against the robustness test (how much accuracy
   each condition really costs).
5. **Real degradations, not only injected ones**: occlude an arm with an object, walk partly out of frame, dim the
   lights, throttle the Wi-Fi. The injected faults are controlled and repeatable, but a real one is more convincing.
6. **IMU** (the paper is skeleton plus IMU). A phone strapped to the wrist can stream accelerometer and gyroscope
   (Android SensorManager) next to the camera. This needs a model trained on 2D pose plus wrist IMU (for example
   UTD-MHAD, right-wrist sensor, 50 Hz, after projecting its 3D skeletons to 2D) and an IMU mode in SafeSense Cam.
   `SafeSenseCOCO` already has an IMU input that is masked today. Simulated IMU must be labelled "simulated" on
   screen if used at all.
7. **Speed**: the action step takes about 100 ms per update with the comparison line on in the sandbox test (two
   model passes plus cleaning). Profile on the laptop; `clean_window` is plain NumPy loops over joints.
8. **Two people**: `track()` keeps people in slots by nearest centre; crossings can still swap them. The demo is
   designed for one person.
9. **Weak classes**: object and fine hand actions (phone, tablet, magic cube) are near 10% for SafeSense-Pose; keep
   them out of the demo list.

---

## 7. Where things came from

- Training data: NTU RGB+D 120, 2D HRNet keypoints as released by PYSKL (`ntu120_hrnet.pkl`), cross-subject split.
- Pose estimation: RTMPose through rtmlib (Body mode: YOLOX detector + RTMPose, ONNX on CPU).
- Model research with 48 linked sources: `docs/robust_har_research.md`.
- All other links: `REFERENCES.md`.
- Git: the code lives in the SafeSense repository (Ashok-19/SafeSense on GitHub), folders `demo/` and `camapp/`,
  branch `claude/project-thread-2g6qi8`. Ask Ashok for access to push changes there.
