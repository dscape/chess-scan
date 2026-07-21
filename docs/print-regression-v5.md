# V5 photographed-workbook regression recovery

Date: 2026-07-21
Reference source SHA-256: `911b4f19be1de1133e66c1a012104d5c4dbf35cef376727a396c851249e8ac2b`

The operator-supplied source image is not stored in Git. The external `print-regressions-v1` corpus retains one consented rectified crop and is hash-described by `benchmarks/print-regression-corpus.json`.

## Diagnosis

Automatic geometry was correct. The source was detected by the contour path with corners `(169,116)`, `(889,19)`, `(952,812)`, and `(81,817)`. Running every historical artifact on the same 512 px rectification isolated the change to v4 inference:

| Model | Correct squares |
|---|---:|
| Argus v2r5 | 60 / 64 |
| Chess Steps v1 | 64 / 64 |
| Chess Steps v1r1 | 64 / 64 |
| Chess Steps v2 | 64 / 64 |
| Chess Steps v3 | 64 / 64 |
| Chess Steps v4 | 62 / 64 |
| Chess Steps v5 | 64 / 64 |

V4 changed the black rook on c8 to a white rook and the black king on e8 to a white king. Its raw probabilities were:

| Square | Expected | V4 prediction | Prediction probability | Expected probability |
|---|---|---|---:|---:|
| c8 | black rook | white rook | 68.43% | 27.63% |
| e8 | black king | white king | 86.10% | 2.74% |

The existing synthetic photo gates did not expose the gap: v4 remained 934/934 exact on the broad synthetic-halftone official retention slice and 12/12 on every enforced photo transformation. On the real photograph, a small blur or a 128–192 px resize round trip restored v4 to 64/64. That isolates the regression to sensitivity to the photograph's sharp paper/print texture, not board geometry or a chess-rule issue.

A runtime blur was rejected. The blur required to fix the workbook reduced the deterministic platform-camera gate from 980 to 848 exact boards. V5 instead adapts all feature blocks from v4 with the consented print crop while replaying platform, official-print, Argus, and synthetic examples.

## Promotion gates

| Gate | V4 | V5 |
|---|---:|---:|
| Reference workbook | 62 / 64 squares | 64 / 64 squares |
| Reference rectified-crop variants | 2 / 8 exact | 8 / 8 exact |
| Clean platform boards | 982 / 996 exact | 987 / 996 exact |
| Camera platform boards | 980 / 996 exact | 980 / 996 exact, +1 square |
| Argus held-out | 4,495 / 4,500 | 4,496 / 4,500 |
| Synthetic replay | 19,378 / 19,500 | 19,382 / 19,500 |
| Official online | 267 / 267 exact | 267 / 267 exact |
| Broad official clean retention | 933 / 934 exact | 934 / 934 exact |
| Real Take Take Take holdouts | 2 / 2 exact | 2 / 2 exact |

Every paired per-platform and per-class gate passed. The artifact/checkpoint equivalence check has a maximum logit difference of `3.814697265625e-06`.

## Feedback correction and future prevention

Two consented local confirmations had accepted v4's invalid two-white-king/no-black-king output unchanged. Their confirmations remain immutable; append-only adjudications now resolve both to the visually verified c8 black rook and e8 black king before export or training. A third confirmation already contained those corrections.

The current confirmation flow rejects illegal king counts before creating feedback. Future automatic candidates must also pass `evaluate_print_regressions.py`; the production learner treats a missing print corpus as an unavailable benchmark rather than skipping the gate.
