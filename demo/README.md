# SafeSense live demo

Phone camera (Android, IP Webcam app) to RTMPose (17 COCO joints, rtmlib on CPU) to SafeSense-Pose (trained on NTU-120) to a browser dashboard on the laptop.

1. In a venv: `pip install -r requirements.txt`. The first run downloads the RTMPose ONNX models from OpenMMLab.
2. Keep `safesense_pose_ntu120.pt` next to `app.py`.
3. On the phone: install IP Webcam, join the same Wi-Fi as the laptop, tap "Start server" and note the URL (for example http://192.168.1.23:8080).
4. Run `python app.py --source http://192.168.1.23:8080/video` and open http://localhost:8000.

Other sources: `--source 0` (laptop webcam), `--source backup.mp4` (recorded video, loops), `--mock` (synthetic skeleton, no camera).
Options: `--pose balanced` (more accurate, slower), `--threshold 0.45`, `--window 3.0`, `--actions "falling,sitting down,standing up"`.

Camera placement: whole body in frame, 2 to 4 m away, phone in landscape.
