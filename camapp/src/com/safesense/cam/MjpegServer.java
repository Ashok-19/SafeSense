package com.safesense.cam;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.Charset;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Tiny HTTP server, compatible with the IP Webcam addresses:
 *   /video (also /videofeed, /mjpeg)  MJPEG stream (multipart/x-mixed-replace), what OpenCV and browsers read
 *   /shot.jpg                          the latest single frame
 *   /status.json                       frame rate, size and connected viewers
 *   /                                  a page showing the stream
 * Frames are only JPEG-encoded while someone is watching, so an idle phone does little work.
 */
public class MjpegServer {
    public static final String BOUNDARY = "safesenseframe";
    private static final Charset ASCII = Charset.forName("US-ASCII");

    private final int port;
    private ServerSocket server;
    private Thread acceptThread;
    private volatile boolean running;

    private final Object frameLock = new Object();
    private byte[] jpeg;
    private long seq;
    private final AtomicInteger viewers = new AtomicInteger();
    private final AtomicInteger shotWaiters = new AtomicInteger();
    public volatile String statusJson = "{}";

    public MjpegServer(int port) { this.port = port; }

    public int viewers() { return viewers.get(); }

    /** True when a frame should be encoded: someone is streaming or waiting for a snapshot. */
    public boolean wanted() { return viewers.get() > 0 || shotWaiters.get() > 0; }

    public void publish(byte[] data) {
        synchronized (frameLock) { jpeg = data; seq++; frameLock.notifyAll(); }
    }

    public synchronized void start() throws IOException {
        if (running) return;
        server = new ServerSocket();
        server.setReuseAddress(true);
        server.bind(new InetSocketAddress(port));
        running = true;
        acceptThread = new Thread(new Runnable() {
            public void run() {
                while (running) {
                    try {
                        final Socket s = server.accept();
                        Thread t = new Thread(new Runnable() { public void run() { handle(s); } }, "http-client");
                        t.setDaemon(true);
                        t.start();
                    } catch (IOException e) {
                        if (!running) break;
                    }
                }
            }
        }, "http-accept");
        acceptThread.setDaemon(true);
        acceptThread.start();
    }

    public synchronized void stop() {
        running = false;
        try { if (server != null) server.close(); } catch (IOException ignored) { }
        synchronized (frameLock) { frameLock.notifyAll(); }
    }

    private void handle(Socket s) {
        try {
            s.setTcpNoDelay(true);
            s.setSoTimeout(10000);
            BufferedReader in = new BufferedReader(new InputStreamReader(s.getInputStream(), ASCII));
            String line = in.readLine();
            if (line == null) { s.close(); return; }
            String[] parts = line.split(" ");
            String path = parts.length > 1 ? parts[1] : "/";
            int q = path.indexOf('?');
            if (q >= 0) path = path.substring(0, q);
            while ((line = in.readLine()) != null && line.length() > 0) { /* skip headers */ }
            s.setSoTimeout(0);
            OutputStream out = s.getOutputStream();
            if (path.equals("/video") || path.equals("/videofeed") || path.equals("/mjpeg") || path.equals("/video.mjpg")) {
                stream(out);
            } else if (path.equals("/shot.jpg") || path.equals("/photo.jpg")) {
                shot(out);
            } else if (path.equals("/status.json")) {
                send(out, "200 OK", "application/json", statusJson.getBytes(ASCII));
            } else if (path.equals("/") || path.equals("/index.html")) {
                String html = "<!doctype html><html><head><meta name=viewport content='width=device-width'><title>SafeSense Cam</title></head>"
                        + "<body style='margin:0;background:#111;color:#ddd;font-family:Arial'><div style='padding:8px'>SafeSense Cam: "
                        + "stream at /video, single frame at /shot.jpg</div><img src='/video' style='width:100%'></body></html>";
                send(out, "200 OK", "text/html; charset=utf-8", html.getBytes(Charset.forName("UTF-8")));
            } else {
                send(out, "404 Not Found", "text/plain", "not found".getBytes(ASCII));
            }
        } catch (IOException ignored) {
        } finally {
            try { s.close(); } catch (IOException ignored) { }
        }
    }

    private void send(OutputStream out, String status, String type, byte[] body) throws IOException {
        String h = "HTTP/1.0 " + status + "\r\nContent-Type: " + type + "\r\nContent-Length: " + body.length
                + "\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
        out.write(h.getBytes(ASCII));
        out.write(body);
        out.flush();
    }

    private void stream(OutputStream out) throws IOException {
        viewers.incrementAndGet();
        try {
            out.write(("HTTP/1.0 200 OK\r\nServer: SafeSenseCam\r\nConnection: close\r\nCache-Control: no-cache, private\r\n"
                    + "Pragma: no-cache\r\nAccess-Control-Allow-Origin: *\r\n"
                    + "Content-Type: multipart/x-mixed-replace; boundary=" + BOUNDARY + "\r\n\r\n").getBytes(ASCII));
            out.flush();
            long last = -1;
            while (running) {
                byte[] f;
                synchronized (frameLock) {
                    long until = System.currentTimeMillis() + 5000;
                    while (running && seq == last) {
                        long left = until - System.currentTimeMillis();
                        if (left <= 0) break;
                        try { frameLock.wait(left); } catch (InterruptedException e) { return; }
                    }
                    if (seq == last || jpeg == null) continue;   // no new frame for 5 s: keep waiting
                    f = jpeg; last = seq;
                }
                out.write(("--" + BOUNDARY + "\r\nContent-Type: image/jpeg\r\nContent-Length: " + f.length + "\r\n\r\n").getBytes(ASCII));
                out.write(f);
                out.write("\r\n".getBytes(ASCII));
                out.flush();
            }
        } finally {
            viewers.decrementAndGet();
        }
    }

    private void shot(OutputStream out) throws IOException {
        shotWaiters.incrementAndGet();
        byte[] f;
        try {
            synchronized (frameLock) {
                long start = seq, until = System.currentTimeMillis() + 3000;
                while (running && seq == start) {
                    long left = until - System.currentTimeMillis();
                    if (left <= 0) break;
                    try { frameLock.wait(left); } catch (InterruptedException e) { break; }
                }
                f = jpeg;
            }
        } finally {
            shotWaiters.decrementAndGet();
        }
        if (f == null) send(out, "503 Service Unavailable", "text/plain", "camera not running".getBytes(ASCII));
        else send(out, "200 OK", "image/jpeg", f);
    }
}
