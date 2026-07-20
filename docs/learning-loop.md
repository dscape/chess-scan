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

Only explicit confirmations are training labels. Intermediate taps are not. Unchanged confirmations are useful positive examples, but should be monitored for inattentive acceptance. A later manual review never mutates the confirmation: it appends an immutable adjudication, and exports and training resolve the newest adjudicated labels.

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

`chess-steps-v2` corrected the nine audited queen-color errors and established the print/geometry gates, but a later provenance audit found that its 19,500-square "Argus replay" was synthetic-only. The original Argus training distribution also contained `chess_positions` and real overlay crops. On the reconstructed held-out split, v2 had fallen from Argus v2r5's 99.91% to 94.51%, dominated by black pieces becoming their matching white classes.

`chess-steps-v3` restored representative Argus coverage. It starts from v2, performs class-weighted supervised recovery over all 80,000 external `chess_positions/train` boards, and retains v2 behavior by distilling augmented official training boards.

`chess-steps-v4` is the active registry revision. It expands the classifier from 315 KB to 2.6 MB after the smaller network demonstrably underfit clear platform glyphs. A domain-balanced external corpus covers 38 Chess.com, 40 Lichess, and five Take Take Take visible piece styles, with 12 globally unseen positions per style. V4 reaches 99.97% square accuracy, 99.90% occupied-square accuracy, and 982/996 exact boards on clean platform holdouts. A deterministic camera/display gate reaches 980/996 exact boards. It remains exact on every official print/photo gate, improves held-out Argus without regressing any class, preserves synthetic replay, classifies two real Take Take Take app holdouts exactly, and classifies the latest corrected Chess.com Glass board exactly. Intentionally concealed themes are excluded because the required identity or color is absent from the pixels.

The external corpora are hash-described by `benchmarks/argus-training-corpus.json` and `benchmarks/platform-training-corpus.json`, mounted read-only in production, and never stored in Git. Every future feedback candidate interleaves labeled Argus and platform replay with user-feedback training and must not regress on either paired gate. Bootstrap upgrades obsolete base registrations through `chess-steps-v3`; feedback-promoted candidates remain active.

## Continuous improvement, not per-request online learning

The application collects feedback continuously and advances one automatic learning cycle at a time:

1. Collect an initial batch of 100 consented boards, then batches of at least 40.
2. Snapshot accepted replay feedback plus the new pending batch.
3. Split at image and installation level, never randomly by square.
4. Initialize from the active ONNX artifact and fine-tune with extra weight on explicitly corrected squares.
5. Reject candidates that regress on the grouped feedback gate, held-out Argus/platform gates, or immutable official online/photo gates.
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
- no overall, occupied-square, or per-class regression on held-out `chess_positions`, and no synthetic replay regression
- no clean or camera/display regression for Chess.com, Lichess, or Take Take Take
- a verified immutable artifact hash

Candidates retain the active classifier architecture and ONNX runtime. The active wide model remains only 2.6 MB, and immutable replay gates bound later feedback training independently of public labels.

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
