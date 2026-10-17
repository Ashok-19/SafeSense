# SafeSense live demo

Android phone camera -> Wi-Fi -> laptop: RTMPose finds 17 body joints -> original SafeSense (trained on NTU-120) labels the last 3 seconds -> dashboard in the browser.

## 1. One-time setup on the laptop

```bash
mkdir -p ~/Desktop/final-year-project/live-demo && cd ~/Desktop/final-year-project/live-demo
# copy these files here: app.py, dashboard.html, safesense_orig.py, safesense_pose.py, requirements.txt
# and the trained model from the Kaggle output of notebook 06: safesense_orig_ntu120.pt
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CPU is enough: the action model takes about 10 to 20 ms per prediction. The first real run downloads the RTMPose models (from download.openmmlab.com) once.

Check the install without a camera:

```bash
python app.py --mock
```

Open http://localhost:8000. A synthetic stick figure should appear, and the header should say "Model: SafeSense (original)". Stop it with Ctrl+C.

## 2. Phone (Android)

1. Install **IP Webcam** (by Pavel Khlebovich) from the Play Store.
2. Connect the phone to the same Wi-Fi as the laptop. A phone hotspot with the laptop joined to it also works and avoids college Wi-Fi blocking devices from seeing each other.
3. In the app: Video preferences > Video resolution 640x480 (enough, and faster). Then tap **Start server** at the bottom.
4. The screen shows an address like `http://192.168.1.23:8080`. Open it in the laptop browser once to check the phone is reachable.

## 3. Run the demo

```bash
cd ~/Desktop/final-year-project/live-demo && source .venv/bin/activate
python app.py --source http://192.168.1.23:8080/video
```

Open http://localhost:8000 (full screen with F11 for the presentation).

The dashboard shows the live video with the skeleton, the current action with its top 5 confidences, a log of recent actions, a red alert banner for falling or staggering, and FPS and timing stats.

## 4. Camera placement and what to perform

- Phone in landscape, on a stand or leaning on something at chest height, 2.5 to 4 m away, whole body in frame from head to feet.
- Front or slightly side-on view, good light, one person in front of the camera.
- Hold each action for 2 to 3 seconds; the label updates 4 times a second and settles in about a second.
- Strongest actions (above 95% on NTU test people): sitting down, standing up, jump up, squat down, falling, pickup, cross arms, arm circles, hands up. Also good: kicking, running on the spot, staggering, hand waving, clapping, drink water, phone call, touch head, touch chest.
- For the fall, use a mat and fall sideways slowly; the alert stays on screen for 6 seconds and can be cleared with the Clear button.

## 5. Options and backup

| Option | What it does |
|---|---|
| `--source 0` | laptop webcam instead of the phone |
| `--source backup.mp4` | play a recorded video in a loop (record one on the phone beforehand as a backup) |
| `--pose balanced` | more accurate joints, slower on CPU |
| `--threshold 0.45` | raise (0.6) if it shows wrong actions, lower (0.35) if it says "no clear action" too often |
| `--actions "falling,sitting down,standing up,squat down"` | limit the dashboard to the actions you will perform |
| `--checkpoint safesense_pose_ntu120.pt` | use the 682k SafeSense-Pose model instead |

## 6. Pretrained PYSKL ST-GCN++ (strongest model)

ST-GCN++ from PYSKL (Apache-2.0) with its official NTU-120 cross-subject weights for 2D HRNet skeletons: 84.4% (joint), 84.8% (bone), 86.4% (joint + bone), about 1.4M parameters. Same 17 COCO joints as the rest of the demo, no training needed.

```bash
bash get_stgcnpp.sh                       # once: downloads stgcnpp_j.pth and stgcnpp_b.pth
bash run.sh http://192.168.1.23:8080 --stgcnpp
```

It looks at the last 3.3 seconds at the camera's real speed, the way it was trained. Joint + bone runs two networks, so each prediction is slower than SafeSense; if the laptop struggles, delete `stgcnpp_b.pth` to run the joint stream only.

## 7. If something goes wrong

- **"model file not found"**: `safesense_orig_ntu120.pt` is not next to `app.py`.
- **Camera FPS stays 0**: the laptop cannot reach the phone. Open the phone URL in the browser; if it does not load, use the phone hotspot.
- **Pose FPS below about 8**: keep `--pose lightweight`, lower the phone resolution, close other programs.
- **"No person in view"**: step back until the whole body is visible.
