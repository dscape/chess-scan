# Chess Scan

Photograph one Chess Steps workbook diagram, correct the model's reading, and open the exact position in Lichess.

Chess Scan extracts the useful single-image classifier from [Argus](https://github.com/dscape/argus) without carrying over its video, physical-board, or VLA pipelines. A small ONNX model reads 64 rectified squares on CPU. Human confirmation is stored as versioned training feedback.

## Product flow

1. Open the in-app camera and frame one diagram.
2. Watch the projected 8×8 grid align; three stable detections lock it green, or use the manual shutter for a faint board.
3. See a brief rectified “64 squares in place” confirmation.
4. Adjust four corners if automatic detection was uncertain and heed low-resolution or king-count warnings.
5. Correct any piece on an editable board.
6. Select side to move and orientation.
7. Save the confirmed position and open Lichess analysis.
8. Use confirmed crops and labels in a gated model-learning cycle.

## Development

Requires Python 3.12, Node.js 22+, and `uv`.

```bash
make install
make dev
```

- Web: http://localhost:5173
- API: http://localhost:8000
- API docs: http://localhost:8000/api/docs

Run all checks and the reproducible model gates:

```bash
make check
make qa-argus
make qa-platform
make qa-online
make qa-stress
```

## Production

The Docker image builds the React frontend and serves it from FastAPI:

```bash
docker compose up --build
# http://localhost:8000
```

Persist `/app/data`; it contains SQLite records, confirmed rectified crops, and promoted candidate artifacts. Browsers require HTTPS for camera access outside `localhost`.

Production runs an automatic learner against the same persistent volume:

```bash
docker compose --profile learning up -d learner
```

The learner waits for 100 initial consented boards, trains a supervised candidate, and runs the immutable official online, print-photo, Argus, and platform gates. A passing candidate remains hidden while it is compared with the active model on at least 40 later confirmations from diverse installations and diagrams. It is promoted only when it makes strictly fewer square errors without regressing exact boards or occupied squares. Rejected training batches are quarantined automatically; successful batches enter the accepted replay pool. Later cycles begin after 40 new boards.

The app checks the SQLite model registry on each scan and atomically reloads a newly promoted ONNX artifact without a redeploy. Model artifacts and lifecycle state remain in the persistent data volume across application deployments. Successful `main` CI dispatches the exact source SHA to `s46-infra`; the repository requires the `S46_INFRA_DISPATCH_TOKEN` Actions secret for that one-time deployment setup.

## Base-model adaptation

The active `chess-steps-v4` model uses a 2.6 MB square CNN trained for standard Lichess, Chess.com, and Take Take Take artwork. Its external `platforms-v1` corpus contains 3,984 FEN-labelled boards spanning 83 visible piece styles, with the final 12 positions per style held out. Intentionally concealed themes—Chess.com Blindfold and Lichess Mono/Disguised—are excluded because their missing pixel information cannot be classified.

The labeled image corpora are deliberately outside Git. Prepare or sync the verified copies, then reproduce the model and run every gate:

```bash
make prepare-argus-data     # ~/chess-scan-training/argus-2026-03-29
make prepare-platform-data  # ~/chess-scan-training/platforms-v1 after acquiring assets
make train-platform-model
make qa-argus
make qa-platform
make qa-online
make qa-stress
```

Set `CHESS_SCAN_ARGUS_DATA_DIR` and `CHESS_SCAN_PLATFORM_DATA_DIR` to use other external locations. [`benchmarks/argus-training-corpus.json`](benchmarks/argus-training-corpus.json) and [`benchmarks/platform-training-corpus.json`](benchmarks/platform-training-corpus.json) hash-describe both corpora without redistributing source images. Production mounts a read-only server copy for automatic replay and gating; `s46-infra/hetzner/scripts/sync-chess-training-data.sh` explicitly pushes or pulls it with rsync.

V4 reaches 99.97% square and 99.90% occupied-square accuracy across the grouped clean platform holdout, with 982/996 exact boards. Its deterministic camera/display gate reaches 99.94% square and 99.80% occupied-square accuracy with 980/996 exact boards. Chess.com is 454/456 exact, Lichess 468/480, and Take Take Take 60/60 on clean captures. Two separate real Take Take Take app boards remain exactly held out. The latest confirmed Chess.com Glass board is 64/64 after entering the immutable training pool.

Official source files are downloaded during training or QA and are not redistributed. V4 remains exact on all 267 reproducible online boards and every enforced print/photo gate, improves the Argus held-out sample from v3's 4,482/4,500 to 4,495/4,500 without regressing any class, and preserves synthetic replay at 19,378/19,500.

## Learning loop

Corrections are exact labels, so the production learner uses supervised fine-tuning initialized from the active ONNX artifact. The normal lifecycle is fully automatic:

```bash
uv sync --extra dev --extra ml
uv run python scripts/automatic_learner.py --once
```

In production the learner polls continuously. Code deployments and model promotions are independent: successful `main` CI dispatches the exact source SHA to `s46-infra`, while a promoted model is activated directly from the persistent registry.

Manual tools remain available for audits and experiments:

```bash
uv run python scripts/export_feedback.py
uv run python scripts/export_preferences.py
uv run python scripts/train_candidate.py --min-boards 100
uv run python scripts/train_candidate.py --min-boards 100 --preference-weight 0.1
uv run python scripts/promote_model.py steps-YYYYMMDDHHMMSS --confirm
```

See [`docs/learning-loop.md`](docs/learning-loop.md) for the distinction between supervised feedback, preference learning, RLHF, evaluation gates, and continuous deployment. The broader localized-corpus, photo-stress, and mobile-usability results are recorded in [`docs/qa-report.md`](docs/qa-report.md).

## Data handling

- Low-resolution live detection frames are processed transiently and never written to storage.
- Final full photographs are deleted after confirmation or after 24 hours.
- Confirmed rectified board crops are retained only when training consent is enabled.
- Every feedback event records the model version that made the prediction.
- Models are immutable and can be rolled back by changing the active registry entry.

## Model provenance

`models/argus-v2r5.onnx` is the MIT-licensed Argus overlay square classifier. `models/chess-steps-v1.onnx` is the original print adaptation, `chess-steps-v2.onnx` is the queen-color revision, `chess-steps-v3.onnx` restores Argus coverage, and `chess-steps-v4.onnx` is the active multi-platform model. Trainable checkpoints and metadata are stored alongside the immutable weight artifacts.

Recorded king-location results:

- Argus baseline: 9/1,064 official positions exact.
- Chess Steps v2: 1,064/1,064 official positions exact after adjudication.
- Independent manually audited gate: 12/12 positions exact.
- Manually audited full-board page: 12/12 boards exact.
- Official interactive source-FEN examples: 204/204 boards exact.
- Official German manual samples: 63/63 standard boards exact after one adjudication.
- Combined additional standard online source images: 1,243/1,243 after adjudication.
- Expanded reminders/manuals: 1,366/1,366 exact kings and 1,364/1,366 board-exact.
- Perspective sample: exact expected FEN, including both kings.
