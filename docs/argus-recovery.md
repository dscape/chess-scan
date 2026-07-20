# Argus data recovery and v3 promotion

- Date: 2026-07-20
- Model: `chess-steps-v3`
- Artifact: `e212ae1e051acf95ffe58f3f743e82b038c7da3c74b46e53676bf4b08db0fa5b`

## Root cause

The replay array used by the original Chess Steps adaptation contains 19,500 balanced synthetic squares. It does not contain the `chess_positions` or real-overlay distributions used by Argus v2r5. The synthetic replay gate passed while the model forgot digital black-piece appearance.

On the reconstructed seed-42 `chess_positions/test` sample, v2 scored 4,253/4,500 (94.51%) and 2,753/3,000 occupied squares (91.77%). Most failures preserved piece type but changed black to white.

## External corpus

The relevant labeled data from `argus_data.tar.gz` is stored outside Git:

- 80,000 `chess_positions/train` boards;
- 20,000 `chess_positions/test` boards;
- 13 `chess_positions/test_real` digital/broadcast crops;
- 19,500 synthetic replay squares and labels.

The verified local default is `~/chess-scan-training/argus-2026-03-29`. Production uses `/opt/s46/chess/src/training-data/argus-2026-03-29`, mounted read-only at `/app/training-data/argus-2026-03-29` in the learner. The committed manifest contains only counts and hashes: [`benchmarks/argus-training-corpus.json`](../benchmarks/argus-training-corpus.json).

Prepare or verify the local corpus with:

```bash
make prepare-argus-data
```

The infra sync command has explicit directions and does not delete either side:

```bash
./hetzner/scripts/sync-chess-training-data.sh push
./hetzner/scripts/sync-chess-training-data.sh pull
```

`pull` is the normal safety-backup operation from production to the operator's computer.

## Training

V3 starts from v2 and uses three stages:

1. balanced recovery using 12,600 deterministic `chess_positions/train` squares plus all 19,500 synthetic replay squares;
2. a second compact stage after resetting the optimizer;
3. one low-learning-rate pass over every square in all 80,000 training boards.

The full pass uses class-weighted loss rather than dropping 4.3 million empty-square observations. At every stage, the current v2 model acts as a retention teacher on 934 augmented official training boards. The held-out official pages and the entire `chess_positions/test` split never enter training.

## Promotion results

| Gate | v2 | v3 |
|---|---:|---:|
| Held-out `chess_positions` | 4,253/4,500 | **4,482/4,500** |
| Held-out occupied squares | 2,753/3,000 | **2,982/3,000** |
| Synthetic replay | 19,348/19,500 | **19,378/19,500** |
| Official online exact boards | 267/267 | 267/267 |
| Reviewed feedback exact boards | 4/13 | 4/13 |
| Reviewed feedback squares | 772/832 | **792/832** |
| Reviewed feedback occupied squares | 158/211 | **177/211** |
| Five photographed-screen squares | 269/320 | **291/320** |
| Five photographed-screen occupied squares | 65/115 | **87/115** |

All enforced clean, 256 px, faded, halftone, moderate-perspective, and severe-perspective photo gates pass exactly. Every held-out `chess_positions` class is non-regressing.

The five photographed-screen boards contain only three distinct positions and are a diagnostic, not a sufficient independent benchmark. Future promotion still requires fresh grouped shadow feedback.

## Continuous learner protection

Production candidate training now interleaves the external labeled Argus replay with consented feedback. Candidate eligibility includes a paired active-versus-candidate Argus evaluation. The automatic fixed-QA stage repeats that paired gate before shadowing. The corpus is mounted read-only so learner code cannot mutate its benchmark.
