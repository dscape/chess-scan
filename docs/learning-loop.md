# Human-feedback learning loop

## What this system is

Chess Scan receives unusually strong human feedback: the user supplies the exact final class for every one of 64 squares. That is supervised ground truth, not merely a thumbs-up or a preference.

The production learning order is therefore:

1. **Supervised fine-tuning (SFT):** train square classification directly against confirmed labels.
2. **Preference learning experiment:** make the confirmed board score above the model's rejected board.
3. **Reward modeling / RL only if it beats SFT:** learn an acceptance reward from candidate boards, then optimize a whole-board policy offline.

Calling step 1 “RLHF” would be inaccurate. It is still learning from human feedback, and it is more data-efficient than RLHF for this task.

## Feedback contract

Every confirmation stores:

- immutable feedback id and timestamp
- model version
- rectified board crop
- original 64 predicted labels and probabilities
- final 64 image-order labels
- changed-square count
- orientation and side to move
- random anonymous installation id for leakage-aware grouping and contribution caps
- training consent

Only explicit confirmations are training labels. Intermediate taps are not. Unchanged confirmations are useful positive examples, but should be monitored for inattentive acceptance.

## Preference records

For corrected boards:

- `chosen`: final human-confirmed labels
- `rejected`: original model labels
- `context`: rectified board crop and model version

A simple pairwise objective scores a board with the sum of its square log-probabilities:

```text
score(board | image) = Σ square_log_probability
loss_preference = -log σ(score(chosen) - score(rejected))
```

This resembles direct preference optimization, but it is not full RLHF. The exact-label cross-entropy objective remains stronger because it identifies every corrected action directly.

A later full RLHF experiment would:

1. Generate multiple whole-board candidates.
2. Learn a reward model from accepted/corrected comparisons.
3. Optimize a whole-board candidate policy against that reward.
4. Compare against the SFT baseline on the same fixed holdout.

Do not run exploratory candidate selection against users. Deliberately showing uncertain alternatives would degrade the scanner experience.

## Chess Steps base adaptation

The initial Argus model was accurate on its rendered digital-board validation set but failed on the Chess Steps king glyph, especially the black king. `chess-steps-v1` reconstructs a trainable fused CNN from the deployed Argus ONNX weights and fine-tunes it with:

- 19,500 balanced replay squares recovered from the March 29 Argus backup
- board crops from 24 official Chess Steps sample PDFs
- 12 manually audited FENs used as exact glyph references
- brightness, contrast, affine, blur, and JPEG augmentation

Only diagrams template-matched to exactly one king of each color enter training. `en_lp_2e` is selection-only, and the independent manually audited `en_lp_3` king page is excluded from template construction, training, and checkpoint selection. Official source images are downloaded by the training script and are not stored in the repository.

The adapted weights moved from 9/1,064 to 1,064/1,064 exact king locations on standard official diagrams. The independent 12-position king gate and the manually audited 12-board full-position reference both pass exactly. Argus replay accuracy improved from 97.77% to 99.04%.

`chess-steps-v2` is the active registry revision. It starts from `chess-steps-v1r1`, retains the QA-tested low-contrast geometry and guarded faded-print preprocessing, and fine-tunes queen color using 2,725 hash-located official queen examples plus Argus and king replay. A corrected audit found nine localized white queens that `v1r1` read as black; `v2` fixes all nine without another clean-source change. It passes 267/267 reproducible official interactive and German-manual examples, 1,243/1,243 additional standard online source images after adjudication, and improves the measured Argus replay slice. The source/result/training manifests live under `benchmarks/`. Bootstrap upgrades only obsolete base registrations through `chess-steps-v1r1`; feedback-promoted candidates remain active.

## Continuous improvement, not per-request online learning

The application collects feedback continuously and advances one automatic learning cycle at a time:

1. Collect an initial batch of 100 consented boards, then batches of at least 40.
2. Snapshot accepted replay feedback plus the new pending batch.
3. Split at image and installation level, never randomly by square.
4. Initialize from the active ONNX artifact and fine-tune with extra weight on explicitly corrected squares.
5. Reject candidates that regress on the grouped feedback gate or immutable official online/photo gates.
6. Keep a passing candidate hidden and evaluate it on confirmations created only after training.
7. Score the active and candidate models against the same final labels.
8. Promote automatically only when the candidate saves a meaningful number of square errors while exact boards and occupied-square errors do not regress.
9. Mark the candidate's training batch accepted after promotion, or quarantine it after rejection.
10. Activate the winner atomically and repeat.

This train-on-A, judge-on-later-B design means an individual public label is never trusted as a deployment decision. Bad feedback can produce a bad candidate, but that candidate must still preserve the fixed benchmarks and beat the active model on later, diverse submissions. Contributions are capped per anonymous installation and perceptually duplicate boards count once.

Updating model weights after each confirmation is intentionally forbidden. It would make results irreproducible and expose the live model to accidental labels, abuse, class collapse, and catastrophic forgetting.

## Automatic promotion metrics

The primary product metric is **whole-board exact match**. Square accuracy can conceal unusable boards. An automatic promotion requires:

- a meaningful reduction in total square errors on fresh shadow feedback
- no regression in whole-board exact matches
- no regression in non-empty square errors
- no regression in grouped square accuracy or macro-F1 across all 13 classes
- exact passage of the fixed online examples, king slices, and photo/geometry stress gates
- a verified immutable artifact hash

Candidates retain the same tiny classifier architecture and runtime as the active model, keeping deployment size and latency bounded independently of public feedback.

At 99% independent square accuracy, expected board exact match is only about `0.99^64 = 52.6%`. “QR-like” reliability requires extremely high per-square accuracy and a strong correction interface.

## Data splits and poisoning controls

- Keep all photos of a page or anonymous installation in one split.
- Deduplicate rectified crops with perceptual hashes before weighting them.
- Cap contributions from one installation and repeated positions.
- Keep pending, accepted, and quarantined feedback pools; only a promoted batch joins accepted replay.
- Evaluate candidates on later confirmations that were unavailable during training.
- Require both kings only as an evaluation slice, not as a label-repair rule.
- Keep manually audited benchmarks that never enter training.
- Preserve model version, dataset snapshot, seed, metrics, and artifact hash for every run.

## Privacy

Low-resolution live camera frames are processed transiently for grid detection and are never written to disk. The final full photograph may contain people, handwriting, and location metadata; it is temporary and deleted after confirmation. The training corpus stores only the perspective-rectified diagram crop, prediction, and labels when consent is enabled.
