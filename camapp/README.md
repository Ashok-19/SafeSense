# SafeSense Cam

A minimal Android app that streams the phone camera to the laptop, in place of IP Webcam: no ads, no extra
features. It serves the same addresses as IP Webcam, so the demo works unchanged:

- `http://<phone-ip>:8080/video`: MJPEG stream (what the dashboard and OpenCV read)
- `http://<phone-ip>:8080/shot.jpg`: one frame
- `http://<phone-ip>:8080/`: a page showing the stream, for a quick check in a browser

The screen shows the phone's address. The dashboard finds it by itself (Find my phone), or type it in Setup. Buttons: Stop/Start, front or back
camera, resolution (640x480 default, 960x720, 1280x720) and Dim screen (black screen while streaming; tap to wake).
Frames are turned upright however the phone is placed, portrait or landscape. They are only encoded while the
laptop is watching, and the screen stays on while the app is open (streaming stops if you leave the app).

Install: copy `release/SafeSenseCam.apk` to the phone, open it, and allow installing from that source when asked.

Build (no Android Studio): `sudo apt install aapt dalvik-exchange zipalign apksigner`, put an API 34 `android.jar`
in `sdk/`, then `bash build.sh`. Keep `release.keystore`: updates must be signed with the same key.
