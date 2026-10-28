package com.safesense.cam;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.OrientationEventListener;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;
import java.net.Inet4Address;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * SafeSense Cam: streams the phone camera to the laptop as MJPEG on port 8080, like IP Webcam, with nothing else.
 * Frames are rotated so people are upright however the phone is placed (portrait or landscape).
 */
@SuppressWarnings("deprecation")   // android.hardware.Camera: simplest API that streams preview frames on every phone
public class MainActivity extends Activity implements SurfaceHolder.Callback, Camera.PreviewCallback {
    static final int PORT = 8080;
    static final int[][] SIZES = {{640, 480}, {960, 720}, {1280, 720}};
    static final int QUALITY = 75;

    private final MjpegServer server = new MjpegServer(PORT);
    private final Handler ui = new Handler(Looper.getMainLooper());
    private SurfaceView preview;
    private SurfaceHolder holder;
    private TextView addrText, statusText;
    private Button btnStart, btnCam, btnSize, btnDim;
    private View dimView;

    private volatile Camera camera;
    private int camId = 0, sizeIdx = 0, camOrientation = 90;
    private boolean front, streaming = true, surfaceReady;
    private int prevW, prevH, devRotation = 0;
    private OrientationEventListener orientationListener;
    private WifiManager.WifiLock wifiLock;

    private Thread encoder;
    private final Object encLock = new Object();
    private byte[] pending;                       // preview buffer waiting to be encoded
    private volatile boolean encoderRun;
    private int framesIn, framesOut;
    private float fpsIn, fpsOut;
    private long fpsT = System.currentTimeMillis();
    private int outW, outH;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        buildUi();
        orientationListener = new OrientationEventListener(this) {
            public void onOrientationChanged(int deg) {
                if (deg == ORIENTATION_UNKNOWN) return;
                devRotation = ((deg + 45) / 90 * 90) % 360;
            }
        };
        WifiManager wm = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
        if (wm != null) { wifiLock = wm.createWifiLock(WifiManager.WIFI_MODE_FULL_HIGH_PERF, "safesense-cam"); wifiLock.setReferenceCounted(false); }
    }

    private Button button(String text, View.OnClickListener l) {
        Button bt = new Button(this);
        bt.setText(text); bt.setAllCaps(false); bt.setTextSize(14); bt.setOnClickListener(l);
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
        lp.setMargins(4, 0, 4, 0); bt.setLayoutParams(lp);
        return bt;
    }

    private void buildUi() {
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.BLACK);
        preview = new SurfaceView(this);
        root.addView(preview, new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        holder = preview.getHolder();
        holder.addCallback(this);

        LinearLayout panel = new LinearLayout(this);
        panel.setOrientation(LinearLayout.VERTICAL);
        panel.setBackgroundColor(0xCC10161D);
        panel.setPadding(24, 20, 24, 24);
        addrText = new TextView(this);
        addrText.setTextColor(Color.WHITE); addrText.setTextSize(20);
        statusText = new TextView(this);
        statusText.setTextColor(0xFFB0BEC5); statusText.setTextSize(13); statusText.setPadding(0, 8, 0, 12);
        panel.addView(addrText); panel.addView(statusText);
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        btnStart = button("Stop", new View.OnClickListener() { public void onClick(View v) { toggleStreaming(); } });
        btnCam = button("Front camera", new View.OnClickListener() { public void onClick(View v) { switchCamera(); } });
        btnSize = button(sizeLabel(), new View.OnClickListener() { public void onClick(View v) { nextSize(); } });
        btnDim = button("Dim screen", new View.OnClickListener() { public void onClick(View v) { setDim(true); } });
        row.addView(btnStart); row.addView(btnCam); row.addView(btnSize); row.addView(btnDim);
        panel.addView(row);
        FrameLayout.LayoutParams plp = new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM);
        root.addView(panel, plp);

        dimView = new TextView(this);
        ((TextView) dimView).setText("Streaming. Tap to wake the screen.");
        ((TextView) dimView).setTextColor(0xFF333333); ((TextView) dimView).setGravity(Gravity.CENTER);
        dimView.setBackgroundColor(Color.BLACK);
        dimView.setVisibility(View.GONE);
        dimView.setOnClickListener(new View.OnClickListener() { public void onClick(View v) { setDim(false); } });
        root.addView(dimView, new FrameLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));
        setContentView(root);
    }

    private String sizeLabel() { return SIZES[sizeIdx][0] + "x" + SIZES[sizeIdx][1]; }

    private void setDim(boolean on) {
        dimView.setVisibility(on ? View.VISIBLE : View.GONE);
        WindowManager.LayoutParams lp = getWindow().getAttributes();
        lp.screenBrightness = on ? 0.01f : WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE;
        getWindow().setAttributes(lp);
    }

    // ------------------------------------------------------------------ lifecycle
    @Override
    protected void onResume() {
        super.onResume();
        if (Build.VERSION.SDK_INT >= 23 && checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 1);
            addrText.setText("Allow camera access to start streaming.");
        }
        if (orientationListener.canDetectOrientation()) orientationListener.enable();
        if (wifiLock != null) wifiLock.acquire();
        if (streaming) startAll();
        ui.post(tick);
    }

    @Override
    protected void onPause() {
        super.onPause();
        stopAll();
        orientationListener.disable();
        if (wifiLock != null && wifiLock.isHeld()) wifiLock.release();
        ui.removeCallbacks(tick);
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] res) {
        if (res.length > 0 && res[0] == PackageManager.PERMISSION_GRANTED && streaming) startAll();
        else addrText.setText("Camera permission is needed. Allow it in Settings > Apps > SafeSense Cam.");
    }

    private boolean hasCameraPermission() {
        return Build.VERSION.SDK_INT < 23 || checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    private void startAll() {
        try { server.start(); } catch (Exception e) { addrText.setText("Cannot open port " + PORT + ": " + e.getMessage()); }
        startEncoder();
        openCamera();
    }

    private void stopAll() {
        closeCamera();
        stopEncoder();
        server.stop();
    }

    private void toggleStreaming() {
        streaming = !streaming;
        if (streaming) startAll(); else stopAll();
        btnStart.setText(streaming ? "Stop" : "Start");
    }

    private void switchCamera() {
        if (Camera.getNumberOfCameras() < 2) return;
        closeCamera();
        camId = (camId + 1) % Camera.getNumberOfCameras();
        if (streaming) openCamera();
    }

    private void nextSize() {
        sizeIdx = (sizeIdx + 1) % SIZES.length;
        btnSize.setText(sizeLabel());
        if (camera != null) { closeCamera(); openCamera(); }
    }

    // ------------------------------------------------------------------ camera
    public void surfaceCreated(SurfaceHolder h) { surfaceReady = true; if (streaming) openCamera(); }
    public void surfaceChanged(SurfaceHolder h, int f, int w, int hgt) { }
    public void surfaceDestroyed(SurfaceHolder h) { surfaceReady = false; closeCamera(); }

    private void openCamera() {
        if (camera != null || !surfaceReady || !hasCameraPermission() || Camera.getNumberOfCameras() == 0) return;
        try {
            camera = Camera.open(camId);
            Camera.CameraInfo info = new Camera.CameraInfo();
            Camera.getCameraInfo(camId, info);
            front = info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT;
            camOrientation = info.orientation;
            btnCam.setText(front ? "Back camera" : "Front camera");
            Camera.Parameters p = camera.getParameters();
            Camera.Size best = null;
            int tw = SIZES[sizeIdx][0], th = SIZES[sizeIdx][1];
            for (Camera.Size s : p.getSupportedPreviewSizes()) {   // closest area to the chosen size, preferring the same aspect
                long d = Math.abs((long) s.width * s.height - (long) tw * th) + (s.width * th == tw * s.height ? 0 : 100000);
                if (best == null || d < Math.abs((long) best.width * best.height - (long) tw * th) + (best.width * th == tw * best.height ? 0 : 100000)) best = s;
            }
            p.setPreviewSize(best.width, best.height);
            p.setPreviewFormat(ImageFormat.NV21);
            int[] range = null;
            for (int[] r : p.getSupportedPreviewFpsRange())          // highest frame rate, up to 30
                if (r[1] <= 30000 && (range == null || r[1] > range[1] || (r[1] == range[1] && r[0] > range[0]))) range = r;
            if (range != null) p.setPreviewFpsRange(range[0], range[1]);
            List<String> fm = p.getSupportedFocusModes();
            if (fm != null && fm.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO)) p.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
            if (p.isVideoStabilizationSupported()) p.setVideoStabilization(false);
            camera.setParameters(p);
            prevW = best.width; prevH = best.height;
            camera.setDisplayOrientation(front ? (360 - camOrientation % 360) % 360 : camOrientation);
            camera.setPreviewDisplay(holder);
            int bufSize = prevW * prevH * 3 / 2;
            for (int i = 0; i < 3; i++) camera.addCallbackBuffer(new byte[bufSize]);
            camera.setPreviewCallbackWithBuffer(this);
            camera.startPreview();
        } catch (Exception e) {
            closeCamera();
            addrText.setText("Camera error: " + e.getMessage());
        }
    }

    private void closeCamera() {
        if (camera == null) return;
        try { camera.setPreviewCallbackWithBuffer(null); camera.stopPreview(); } catch (Exception ignored) { }
        camera.release();
        camera = null;
        synchronized (encLock) { pending = null; }
    }

    /** Clockwise rotation that makes the frame upright for how the phone is held right now. */
    private int frameRotation() {
        return front ? (camOrientation - devRotation + 360) % 360 : (camOrientation + devRotation) % 360;
    }

    @Override
    public void onPreviewFrame(byte[] data, Camera cam) {
        framesIn++;
        synchronized (encLock) {
            if (pending == null && server.wanted()) { pending = data; encLock.notify(); return; }
        }
        cam.addCallbackBuffer(data);   // encoder busy or nobody watching: drop this frame
    }

    // ------------------------------------------------------------------ encoding
    private void startEncoder() {
        if (encoder != null) return;
        encoderRun = true;
        encoder = new Thread(new Runnable() {
            public void run() {
                ByteArrayOutputStream bos = new ByteArrayOutputStream(200000);
                byte[] rot = null;
                while (encoderRun) {
                    byte[] data;
                    synchronized (encLock) {
                        while (encoderRun && pending == null) { try { encLock.wait(500); } catch (InterruptedException e) { return; } }
                        data = pending;
                    }
                    if (data == null) continue;
                    Camera c = camera;
                    int w = prevW, h = prevH, r = frameRotation();
                    try {
                        if (data.length != w * h * 3 / 2) continue;   // a buffer from before a size change
                        byte[] src = data;
                        if (r != 0) {
                            if (rot == null || rot.length != data.length) rot = new byte[data.length];
                            rotateNV21(data, rot, w, h, r);
                            src = rot;
                            if (r == 90 || r == 270) { int t = w; w = h; h = t; }
                            giveBack(c, data); data = null;   // the rotated copy is all we need now
                        }
                        bos.reset();
                        new YuvImage(src, ImageFormat.NV21, w, h, null).compressToJpeg(new Rect(0, 0, w, h), QUALITY, bos);
                        server.publish(bos.toByteArray());
                        outW = w; outH = h; framesOut++;
                    } catch (Exception e) {
                        // a frame lost while the camera was being switched; carry on with the next one
                    } finally {
                        synchronized (encLock) { pending = null; }
                        if (data != null) giveBack(c, data);
                    }
                }
            }
        }, "jpeg-encoder");
        encoder.start();
    }

    private void giveBack(Camera c, byte[] buf) {
        if (c == null || c != camera) return;   // that camera is closed: its buffers are gone with it
        try { c.addCallbackBuffer(buf); } catch (Exception ignored) { }
    }

    private void stopEncoder() {
        encoderRun = false;
        synchronized (encLock) { encLock.notifyAll(); }
        if (encoder != null) { try { encoder.join(1000); } catch (InterruptedException ignored) { } }
        encoder = null;
    }

    /** Rotate an NV21 frame w x h clockwise by 90, 180 or 270 degrees into out. */
    static void rotateNV21(byte[] in, byte[] out, int w, int h, int rot) {
        int frame = w * h;
        if (rot == 180) {
            for (int i = 0; i < frame; i++) out[i] = in[frame - 1 - i];
            for (int i = 0; i < frame / 2; i += 2) {
                out[frame + i] = in[frame + frame / 2 - 2 - i];
                out[frame + i + 1] = in[frame + frame / 2 - 1 - i];
            }
            return;
        }
        // 90: new (x', y') = (h - 1 - y, x) in a h x w image; 270: (y, w - 1 - x)
        int i = 0;
        if (rot == 90) {
            for (int x = 0; x < w; x++) for (int y = h - 1; y >= 0; y--) out[i++] = in[y * w + x];
            for (int x = 0; x < w; x += 2) for (int y = h / 2 - 1; y >= 0; y--) {
                out[i++] = in[frame + y * w + x]; out[i++] = in[frame + y * w + x + 1];
            }
        } else {
            for (int x = w - 1; x >= 0; x--) for (int y = 0; y < h; y++) out[i++] = in[y * w + x];
            for (int x = w - 2; x >= 0; x -= 2) for (int y = 0; y < h / 2; y++) {
                out[i++] = in[frame + y * w + x]; out[i++] = in[frame + y * w + x + 1];
            }
        }
    }

    // ------------------------------------------------------------------ status
    private final Runnable tick = new Runnable() {
        public void run() {
            long now = System.currentTimeMillis();
            float dt = (now - fpsT) / 1000f;
            if (dt >= 1f) { fpsIn = framesIn / dt; fpsOut = framesOut / dt; framesIn = framesOut = 0; fpsT = now; }
            List<String> ips = addresses();
            StringBuilder sb = new StringBuilder();
            if (!streaming) sb.append("Stopped. Tap Start to stream.");
            else if (ips.isEmpty()) sb.append("No network. Connect to Wi-Fi or turn on the hotspot.");
            else {
                sb.append("Laptop camera address:");
                for (String ip : ips) sb.append("\n").append(ip);
            }
            if (camera != null || !streaming) addrText.setText(sb.toString());
            int v = server.viewers();
            statusText.setText(String.format("%s camera, %dx%d at %.0f fps | sending %dx%d at %.0f fps to %d viewer%s",
                    front ? "front" : "back", prevW, prevH, fpsIn, outW, outH, fpsOut, v, v == 1 ? "" : "s"));
            server.statusJson = String.format("{\"camera\":\"%s\",\"width\":%d,\"height\":%d,\"fps\":%.1f,\"sending_fps\":%.1f,\"viewers\":%d}",
                    front ? "front" : "back", outW, outH, fpsIn, fpsOut, v);
            ui.postDelayed(this, 1000);
        }
    };

    /** "ip:port (Wi-Fi)" for each IPv4 address a laptop can reach: Wi-Fi and the phone's own hotspot, not mobile data. */
    static List<String> addresses() {
        List<String> out = new ArrayList<String>();
        try {
            for (NetworkInterface ni : Collections.list(NetworkInterface.getNetworkInterfaces())) {
                if (!ni.isUp() || ni.isLoopback()) continue;
                String n = ni.getName().toLowerCase();
                if (n.contains("rmnet") || n.startsWith("ccmni") || n.startsWith("seth") || n.startsWith("wwan") || n.startsWith("pdp")
                        || n.startsWith("dummy") || n.startsWith("tun") || n.startsWith("ppp")) continue;   // mobile data, VPN
                String label = n.startsWith("ap") || n.startsWith("swlan") || n.startsWith("softap") || n.equals("wlan1") ? "hotspot"
                        : n.startsWith("wlan") ? "Wi-Fi" : n;
                for (InetAddress a : Collections.list(ni.getInetAddresses())) {
                    if (!(a instanceof Inet4Address) || a.isLoopbackAddress()) continue;
                    String e = a.getHostAddress() + ":" + PORT + "  (" + label + ")";
                    boolean dup = false;
                    for (String o : out) if (o.startsWith(a.getHostAddress() + ":")) dup = true;
                    if (!dup) out.add(e);
                }
            }
        } catch (Exception ignored) { }
        return out;
    }
}
