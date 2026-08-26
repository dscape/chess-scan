# V6 shadow validation

`chess-steps-v6` remains hidden while `chess-steps-v5` supplies every user-visible label. A confirmation created after the shadow start is paired against both models only after the learner receives the final corrected labels.

## Promotion cohort

The automatic gate requires all of the following:

- at least 40 perceptually distinct confirmed boards;
- at least eight anonymous installations, capped at five boards each;
- for the print specialist, at least ten visually routed boards from three installations;
- a meaningful paired square-error reduction;
- no exact-board or occupied-square regression;
- no installation-macro accuracy, per-class correct-square, macro-F1, individual-class F1, or visual-route regression;
- exact V5/V6 label parity on every board sent through V6's byte-identical general model;
- all immutable online, photo, print, platform, Argus, and runtime-policy gates still passing.

Changing the model, route threshold, or promotion criteria after looking at this cohort makes it development data. A changed candidate must start a new later shadow cohort.

## First-device pilot

Before recruiting the complete cohort, one operator should submit three distinct confirmed positions from the production camera:

1. A pale or diagonally hatched workbook diagram in ordinary indoor light.
2. A different page of that print style with a modest angle or small amount of glare.
3. An ordinary non-hatched print or screen diagram as a routing hard negative.

For each position:

1. Open `https://chess.s46.dev` on the phone; do not select an existing photo.
2. In the camera controls, optionally press **Share clip** beside the threshold and **Photos** options. The single opt-in also permits the corrected board crop to enter training, persists for later scans in that browser, and links to the full data policy. The clip records only the camera, never audio or the browser screen, and expires automatically.
3. Keep the intended diagram dominant in the frame and watch **scan readiness** rise. Verify that the tracked grid surrounds the intended diagram; the percentage estimates capture quality, not label accuracy.
4. Let automatic capture start at the selected Quick, Balanced, or Precise threshold, then hold still through all four progress marks.
5. Record the selected threshold and displayed usable-frame count. Retake if the target is wrong or the image is visibly blurred.
6. Correct every wrong square rather than accepting a plausible-looking position.
7. Choose the actual board orientation and side to move.
8. Confirm the position. There is no second training-consent control; the **Share clip** choice from step 2 governs both retained training crops and camera diagnostics.
9. Do not submit the same page or diagram twice.

After these three confirmations, stop collection. Run the learner once and inspect:

```bash
uv run python scripts/automatic_learner.py --once
make report-capture
make report-shadow
```

The pilot is acceptable when all three automatic captures target the intended board, readiness progresses without amber/green flashing, the diagnostic (when enabled) covers the complete burst, most bursts retain at least three frames, and no new systematic geometry or focus failure appears. Exact model accuracy is reported but is not a pilot pass/fail criterion at only three boards.

## Full cohort

Only after the pilot passes should collection expand. Recruit at least seven additional installations. Each installation should contribute up to five distinct diagrams, with at least three installations contributing genuine diagonal-hatched print. Include ordinary print, screens, varied phones, lighting, modest perspective, and realistic glare so the router is tested on both positives and hard negatives.

Do not coach contributors to accept V5's suggestion, manufacture corrections, repeat one page, or optimize toward a desired V6 result. The final labels must reflect only the photographed diagram.

At 10, 20, and 40 diverse boards, inspect the aggregate reports for capture regressions and data-quality problems without changing V6. Promotion remains automatic only after the full gate passes; otherwise the candidate continues to 120 boards or is rejected.
