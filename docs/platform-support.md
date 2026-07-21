# Platform model and external corpus

`chess-steps-v5` retains v4's normal visible board-theme coverage for Chess.com, Lichess, and Take Take Take while recovering a real photographed-workbook regression. It does not claim to infer information deliberately removed by accessibility or novelty themes.

## External corpus

The operator copy lives at:

```text
~/chess-scan-training/platforms-v1
```

Production uses:

```text
/opt/s46/chess/src/training-data/platforms-v1
```

The consented photographed-print regression corpus lives alongside it as
`~/chess-scan-training/print-regressions-v1` locally and
`/opt/s46/chess/src/training-data/print-regressions-v1` in production. It retains only rectified crops and is verified against `benchmarks/print-regression-corpus.json`.

The platform corpus contains 3,984 generated FEN-labelled boards:

| Platform | Piece styles | Train boards | Test boards |
|---|---:|---:|---:|
| Chess.com | 38 | 1,368 | 456 |
| Lichess | 40 | 1,440 | 480 |
| Take Take Take | 5 | 180 | 60 |

The final 12 positions are globally held out for every style. Two Take Take Take app screenshots are separate real holdouts. Confirmed user crops enter only with training consent.

`benchmarks/platform-training-corpus.json` records source and label-manifest hashes without redistributing any images. Run `scripts/prepare_platform_training_data.py` only after the external asset inventory has been prepared. Preparation validates every curated inventory before rendering, rejects missing or incomplete piece styles, and publishes generated boards and manifests only after the complete render succeeds. Coordinate rendering requires the original hash-verified font rather than selecting a host-specific fallback. On systems without the macOS font path, set `CHESS_SCAN_PLATFORM_FONT` to an operator-managed copy with the expected SHA-256 printed by the preparation command.

The committed manifest's style counts are authoritative. Use `--allow-inventory-change` only when deliberately creating and reviewing a new corpus revision, and pass an explicit new version such as `--corpus-version platforms-v2`. All platform commands respect `CHESS_SCAN_PLATFORM_DATA_DIR`.

### Source handling

- Lichess assets come from its public `lila` source tree. Licenses differ by piece set; retain upstream attribution and license metadata.
- Chess.com assets remain subject to Chess.com's terms. They are kept only in the private operator corpus and are not committed or redistributed.
- Take Take Take assets are rendered from its public web client bundle and remain outside the source repository.
- Chess.com Blindfold and Lichess Mono/Disguised are excluded because piece identity or color is intentionally absent from the image.

## Reproduction

```bash
make prepare-platform-data
make train-platform-model
make train-print-recovery
make qa-platform
make qa-argus
make qa-print
make qa-online
make qa-stress
```

Training is domain-balanced across platform boards, Argus replay, synthetic replay, official print boards, and consented feedback. Replay squares remain memory-mapped and are preprocessed in batches by persistent data-loader workers. Whole-board photometric, compression, display, and perspective round trips are applied before the usual 64-square preprocessing.

Pass `--variant` more than once to evaluate several platform variants in one verified corpus pass. Candidate and baseline inference share each decoded and transformed input:

```bash
python scripts/evaluate_platforms.py \
  --baseline models/chess-steps-v4.onnx \
  --variant clean \
  --variant camera
```

MPS optimization is not byte deterministic. A reproduction must pass every gate independently rather than matching the promoted artifact hash.

## Promotion results

The promoted artifact is `models/chess-steps-v5.onnx`.

| Gate | v4 | v5 |
|---|---:|---:|
| Reference workbook photograph | 62 / 64 | 64 / 64 |
| Clean platform squares | 63,724 / 63,744 | 63,729 / 63,744 |
| Clean platform occupied | 19,651 / 19,671 | 19,656 / 19,671 |
| Clean platform exact boards | 982 / 996 | 987 / 996 |
| Camera/display squares | 63,704 / 63,744 | 63,705 / 63,744 |
| Camera/display exact boards | 980 / 996 | 980 / 996 |
| Argus held-out squares | 4,495 / 4,500 | 4,496 / 4,500 |
| Synthetic replay | 19,378 / 19,500 | 19,382 / 19,500 |
| Official online boards | 267 / 267 | 267 / 267 |
| Real Take Take Take holdouts | 2 / 2 | 2 / 2 |
| Latest Chess.com Glass board | 64 / 64 | 64 / 64 |

V5's clean per-platform exact-board results are 454/456 Chess.com, 473/480 Lichess, and 60/60 Take Take Take. Every included style remains represented and measured rather than silently excluded.

## Automatic learning

The production learner mounts the corpus read-only. Feedback candidates:

1. Load the active wide checkpoint next to its ONNX artifact.
2. Interleave platform and Argus replay with grouped feedback training.
3. Run paired clean and camera platform evaluations and the immutable photographed-print regression.
4. Reject any print, platform, exact-board, occupied-square, Argus, synthetic, or official regression.
5. Continue to fresh shadow feedback only after all immutable gates pass.

Predictions remain raw model outputs. No platform-specific chess-rule repair is applied.
