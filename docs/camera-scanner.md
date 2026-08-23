# Camera scanner

The live scanner is designed as a lock-and-refine pipeline rather than a sequence of full photo scans.

## Fast lock

The browser sends a 720 px JPEG preview to the transient board-detection endpoint. Live detection uses a bounded OpenCV path: normalized checkerboard search, a fast classic check, one enhanced fallback, and a limited contour search. The slower two-image legacy fallback remains reserved for a manually chosen still image and cannot stall the live loop.

A successful live detection returns:

- the projected 8×8 grid;
- one raw 64-square model prediction;
- localized glare estimates; and
- server processing time.

The camera locks the first complete, sufficiently large board. It no longer waits for a second serial network round trip. Provisional pieces appear on the projected grid while the retained scan is prepared.

The accepted source is capped at 1600 px and JPEG quality 0.88. This remains above the classifier's 512 px rectified input while reducing mobile encoding and upload cost.

## Multi-frame refinement

A clean automatic lock submits only the accepted frame. When preview analysis detects localized glare, the camera adds two adjacent video frames for refinement. Supplemental frames are transient and limited by the same upload-body cap as the primary image. The API rejects frames whose dimensions or projected grid no longer agree with the locked geometry.

Every accepted frame is rectified independently. For each of the 64 squares, deterministic image quality weighs specular clipping, local contrast, and sharpness. The clearest square crop is selected into one 512×512 composite board, and the normal ONNX classifier runs once on that image. The runtime still returns raw model predictions; it does not repair king counts or alter labels from chess legality.

This can recover a square when hand movement shifts glare between nearby frames. It cannot recover ink hidden by fixed glare in every frame. Residual glare is stored with the scan, warned at the top of the check screen, and outlined in gold on each affected square for human review.

## Opt-in camera diagnostics

The capture screen offers a separate, unchecked **Share a short camera diagnostic** option. It records only the camera stream:

- no microphone track;
- no browser screen or other applications;
- at most 10 seconds;
- approximately 700 kbit/s;
- uploaded only after the learner confirms the position; and
- deleted automatically after 30 days by default.

Abandoned scans never upload their in-memory recording. If automatic camera locking fails and the learner switches to **Photos**, the opted-in camera clip follows that photo through confirmation so the failure still has corrected labels. Each confirmed position can store at most one diagnostic, bounded to 8 MB. Recordings are not publicly served and are not consumed automatically by model training. The immutable diagnostic record includes a content hash, bounded browser user-agent, capture timings, glare count, and camera dimensions; its confirmed-feedback link supplies the model version, raw prediction, and final corrected labels during export. This makes the clip useful for replaying geometry and latency failures without conflating it with supervised training consent.

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
