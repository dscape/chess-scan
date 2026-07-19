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

Run all checks and the reproducible official-source QA gates:

```bash
make check
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

The learner waits for 100 initial consented boards, trains a supervised candidate, and runs the immutable official online and photo gates. A passing candidate remains hidden while it is compared with the active model on at least 40 later confirmations from diverse installations and diagrams. It is promoted only when it makes strictly fewer square errors without regressing exact boards or occupied squares. Rejected training batches are quarantined automatically; successful batches enter the accepted replay pool. Later cycles begin after 40 new boards.

The app checks the SQLite model registry on each scan and atomically reloads a newly promoted ONNX artifact without a redeploy. Model artifacts and lifecycle state remain in the persistent data volume across application deployments. Successful `main` CI dispatches the exact source SHA to `s46-infra`; the repository requires the `S46_INFRA_DISPATCH_TOKEN` Actions secret for that one-time deployment setup.

## Base-model adaptation

The `chess-steps-v1` weights fine-tune the original Argus ONNX model rather than restarting randomly. The active `chess-steps-v2` revision adds QA-tested geometry/preprocessing and corrects nine adjudicated queen-color errors without regressing the clean gates. Reproducible base training combines the balanced 19,500-square replay set from the March 29 Argus backup with augmented diagrams from 24 official Chess Steps sample PDFs:

```bash
uv run --extra ml --with 'pymupdf>=1.25,<2' \
  python scripts/train_chess_steps_model.py

# Reproduce the queen-color adaptation candidate
uv run --extra ml --with 'pymupdf>=1.25,<2' \
  python scripts/train_queen_adaptation.py
```

Official source files are downloaded during training or QA and are not redistributed. The queen adaptation was reconstructed end-to-end from 167 hash-verified PDFs and 4,893 source locators; its gate metrics matched the promoted artifact despite non-byte-identical MPS training. The recorded gates require exact king locations on an independent manually audited page, exact positions across 1,064 standard official diagrams, 267/267 reproducible interactive and German-manual examples, and no regression on the Argus replay gate. Source and result manifests live under [`benchmarks/`](benchmarks/).

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

`models/argus-v2r5.onnx` is the MIT-licensed Argus overlay square classifier and remains the replay baseline. `models/chess-steps-v1.onnx` is the original domain-adapted weight artifact. `models/chess-steps-v1r1.onnx` records the first QA runtime revision. `models/chess-steps-v2.onnx` is the active immutable queen-color and QA revision; trainable checkpoints and metadata are stored alongside the weight artifacts.

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
