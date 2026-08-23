# Camera scanner

The live scanner is designed as a lock-and-refine pipeline rather than a sequence of full photo scans.

## Fast lock

The browser sends a 720 px JPEG preview to the transient board-detection endpoint. Live detection uses a bounded OpenCV path: normalized checkerboard search, a fast classic check, one enhanced fallback, and a limited contour search. The slower two-image legacy fallback remains reserved for a manually chosen still image and cannot stall the live loop.

A successful live detection returns:

- the projected 8×8 grid;
- one raw 64-square model prediction;
- localized glare estimates; and
- server processing time.

The camera locks the first complete board with enough retained-source detail. The resolution floor is measured after mapping the preview corners back to the source frame, so lowering preview resolution cannot accidentally make auto-lock harder. It no longer waits for a second agreeing network round trip. Provisional pieces stay on the frozen grid while the retained scan is refined.

The accepted source is capped at 1600 px and JPEG quality 0.88. The adaptive detail floor targets at least 384 source pixels across the board—about 48 pixels per square before the model's 64-pixel input—while still supporting lower-resolution cameras.

## Multi-frame refinement

Every camera lock—automatic or manual—captures three adjacent refinement frames over roughly 270 ms. Supplemental frames are transient and limited by the same upload-body cap as the primary image.

The backend first tests the locked geometry, then tracks the 7×7 internal grid with optical flow when the camera has drifted. A bounded independent detection is the fallback. Frames with a different shape, invalid grid, or insufficient overlap with the locked board are rejected. Accepted frames are rectified with their own tracked corners rather than reusing stale coordinates.

For each of the 64 squares, deterministic image quality weighs specular clipping, local contrast, and sharpness. The clearest square crop is selected into one 512×512 composite board, and the normal ONNX classifier runs once on that image. The sharpest accepted full frame is retained for corner adjustment. The runtime still returns raw model predictions; it does not repair king counts or alter labels from chess legality.

This can recover a square when hand movement shifts glare or one frame is motion-blurred. It cannot recover ink hidden or out of focus in every frame. Residual glare is stored with the scan and outlined in gold; low sharpness produces an explicit whole-board warning for human review.

## Opt-in camera diagnostics

The capture screen offers a separate, unchecked **Share a short camera diagnostic** option. It records only the camera stream:

- no microphone track;
- no browser screen or other applications;
- at most 10 seconds;
- approximately 700 kbit/s;
- uploaded only after the learner confirms the position; and
- deleted automatically after 30 days by default.

Abandoned scans never upload their in-memory recording. If automatic camera locking fails and the learner switches to **Photos**, the opted-in camera clip follows that photo through confirmation so the failure still has corrected labels. Each confirmed position can store at most one diagnostic, bounded to 8 MB. The app checks an exact supported video MIME type before enabling consent, includes Safari H.264 MP4 support, waits for the post-confirmation upload, and reports recording or upload failure instead of silently dropping it. Recordings are not publicly served and are not consumed automatically by model training. The immutable diagnostic record includes a content hash, bounded browser user-agent, capture timings, glare count, sharpness, and camera dimensions; its confirmed-feedback link supplies the model version, raw prediction, and final corrected labels during export. This makes the clip useful for replaying geometry and latency failures without conflating it with supervised training consent.

Operators can disable collection or shorten retention:

```bash
CHESS_SCAN_CAPTURE_DIAGNOSTICS_ENABLED=false
CHESS_SCAN_CAPTURE_DIAGNOSTIC_RETENTION_DAYS=14
```

Export an auditable manifest for offline replay before the records expire:

```bash
uv run python scripts/export_capture_diagnostics.py
```

The manifest references files in the persistent data volume; it does not copy videos or extend their database expiry.

## Performance evaluation

Measure from the first usable camera frame, not from page load:

- first detection round trip;
- server detection and preview inference time;
- board-lock latency;
- retained scan round trip;
- false-lock rate;
- exact-board and per-square accuracy; and
- correction rate inside glare-marked squares.

Performance changes must preserve the photographed-print, platform-camera, Argus, and confirmed-feedback gates. Actual p50 and p95 targets should be promoted only after replay and real-device measurements, especially on iOS Safari and constrained mobile networks.
