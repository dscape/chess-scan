# Camera scanner

The live scanner is designed as a select-and-refine pipeline rather than a sequence of surprise automatic captures.

## Explicit target selection

The browser sends a 720 px JPEG preview to the transient board-detection endpoint. Live detection uses a bounded OpenCV path: normalized checkerboard search, a fast classic check, one enhanced fallback, and a limited contour search. The slower two-image legacy fallback remains reserved for a manually chosen still image and cannot stall the live loop.

A successful live detection returns:

- the projected 8×8 grid;
- one raw 64-square model prediction;
- localized glare estimates; and
- server processing time.

The camera never captures the first grid it happens to find. Two consecutive detections must describe the same board before the shutter becomes available. The learner checks that the green projected grid surrounds the intended diagram and explicitly taps **Capture board**. This keeps pages containing several diagrams under human control and gives mobile autofocus time to settle.

The resolution floor is measured after mapping preview corners back to the retained source frame, so lowering preview resolution cannot make the readiness gate harder. Provisional pieces remain projected over the live camera while the learner selects the target.

The accepted source is capped at 1600 px and JPEG quality 0.88. The adaptive detail floor targets at least 384 source pixels across the board—about 48 pixels per square before the model's 64-pixel input—while still supporting lower-resolution cameras.

## Multi-frame refinement

An explicit capture retains the selected frame and then captures three adjacent refinement frames over roughly 540 ms. The live feed remains visible with **1 of 4** through **4 of 4** progress and tells the learner to hold still. Only after the burst is complete does the UI freeze the selected frame and turn off the camera. Supplemental frames are transient and limited by the same upload-body cap as the primary image.

The backend tracks the 7×7 internal grid from the selected frame into each supplemental frame with optical flow. A bounded independent detection is the fallback. It never treats stale primary corners as supplemental geometry merely because they still overlap a grid. Frames with different dimensions, an unavailable grid, or insufficient overlap with the selected board are rejected. The review reports the exact accepted/submitted count, such as **3 of 4 frames usable**.

For each of the 64 squares, deterministic image quality weighs specular clipping, local contrast, and sharpness. The clearest square crop is selected into one 512×512 composite board. The unmodified center-view prediction from that composite remains the presented suggestion. Nine nearby crop predictions and one center prediction from every accepted camera frame measure visual stability; their labels and full probabilities, per-frame quality, rejection reasons, selected-square counts, and agreement scores are stored for audit. Individual supplemental images remain transient. No chess legality, king count, or piece-count rule participates in either the suggestion or its evidence.

Frame composition can recover a square when hand movement shifts glare or one frame is motion-blurred. Stability views do not alter the suggestion. They keep a square outlined when nearby crops or camera frames disagree, even if the center view reports high confidence. Residual glare is stored with the scan and outlined in gold; low sharpness produces an explicit whole-board warning for human review.

A retrospective 98-board production replay found that averaging ±2-pixel whole-board shifts improved photographed-workbook exact matches from 62 to 73, but reduced held-out platform exact matches from 987 to 610 clean boards and from 980 to 657 camera variants out of 996. The average therefore remains evidence only and never replaces the raw center result. Runtime-policy evaluation must pass the same full platform gates as model weights before any future postprocessing can change labels. The 9–13 CPU evidence views add roughly 100–150 ms locally after frame composition. ONNX calls are capped at nine boards so activation memory stays bounded as supplemental frames are added.

## Opt-in camera diagnostics

The capture screen offers a separate, unchecked **Share a short camera diagnostic** option. It records only the camera stream:

- no microphone track;
- no browser screen or other applications;
- at most 10 seconds;
- approximately 700 kbit/s;
- uploaded only after the learner confirms the position; and
- deleted automatically after 30 days by default.

Abandoned scans never upload their in-memory recording. A recording now continues through the visible four-frame burst instead of stopping before the supplemental frames it is meant to audit. If live detection fails and the learner switches to **Photos**, the opted-in camera clip follows that photo through confirmation so the failure still has corrected labels. Each confirmed position can store at most one diagnostic, bounded to 8 MB. The app checks an exact supported video MIME type before enabling consent, includes Safari H.264 MP4 support, waits for the post-confirmation upload, and reports recording or upload failure instead of silently dropping it. Recordings are not publicly served and are not consumed automatically by model training. The immutable diagnostic record includes a content hash, bounded browser user-agent, capture timings, glare count, sharpness, and camera dimensions; its confirmed-feedback link supplies the model version, raw prediction, and final corrected labels during export. This makes the clip useful for replaying geometry and latency failures without conflating it with supervised training consent.

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
- stable-board readiness and explicit-capture latency;
- retained scan round trip;
- wrong-target and discarded-frame rate;
- exact-board and per-square accuracy; and
- correction rate inside glare-marked squares.

Performance changes must preserve the photographed-print, platform-camera, Argus, and confirmed-feedback gates. Actual p50 and p95 targets should be promoted only after replay and real-device measurements, especially on iOS Safari and constrained mobile networks.
