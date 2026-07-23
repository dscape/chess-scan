# Chess Scan

Photograph one chess diagram, correct the model's reading, and review the exact position on an interactive board.

Chess Scan extracts the useful single-image classifier from [Argus](https://github.com/dscape/argus) without carrying over its video, physical-board, or VLA pipelines. A small ONNX model reads 64 rectified squares on CPU. Human confirmation is stored as versioned training feedback. Local Stockfish analysis and deterministic evidence evaluators turn one corrected position into an independently branded educational review; Lichess remains available for advanced analysis.

## Product flow

1. Open the in-app camera and frame one diagram.
2. Watch the projected 8×8 grid align; two agreeing detections capture the exact identified frame, or use the manual shutter for a faint board.
3. Check the rectified crop immediately and adjust four corners when automatic framing was uncertain.
4. Heed low-resolution or king-count warnings and correct any piece on the editable board.
5. Select side to move and orientation.
6. Save the confirmed position and read its detected topic and spoiler-free human-written hint.
7. Try legal moves on the full board while the local evaluation bar updates.
8. After the first move, inspect grounded arrows and text annotations or open advanced Lichess analysis.
9. Use confirmed crops and labels in a gated model-learning cycle.

## Development

Requires Python 3.12, Node.js 22+, and `uv`. Rebuilding the external Lichess benchmark also requires `zstd`.

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
make qa-print
make qa-review
make qa-commentary
make qa-coaching
make qa-online
make qa-stress
```

## Position reviews

The review pipeline keeps separate responsibilities:

- [`chess.js`](https://github.com/jhlywa/chess.js) validates every move made on the browser board.
- Stockfish calculates three stable candidate lines with WDL evidence locally in a browser Worker.
- After the learner moves, Stockfish force-searches that attempt so equivalent moves are accepted and mistakes receive a concrete hypothetical refutation.
- The API validates every move, normalizes score point of view, and emits versioned evidence with actors, targets, causal move endpoints, proof type, scope, and explicit hypothetical-line roles.
- Deterministic tactical detectors are gated against two disjoint, balanced 1,000-puzzle Lichess sets. Held-out accuracy is 99.8% with official setup evidence; FEN-only accuracy is 99.78% across the nine nonhistorical motifs. Mapped-claim precision is 99.77% with setup evidence and 100% in both FEN-only and production-planner modes.
- Human-authored copy verbalizes only selected evidence. Every annotation cites evidence IDs, and a deterministic fallback abstains when no causal subject is proven.
- Evidence-backed board stories distinguish the learner's move, engine choice, hypothetical reply, tactical rays, targets, and motif badges without asking the frontend or a model to reason about chess.
- An optional, separately loaded coaching layer can ask an OpenAI-compatible model to order one or two pre-verified claim IDs. It is disabled by default, cannot write chess claims or alter Stockfish, and never receives images, record IDs, user text, or article annotations.
- Review runs, helpful/problem reports, and expert adjudications are append-only, preserving the engine, contract, output, and required regression fixture for each approved fix.
- A checksum-pinned expert-commentary benchmark compares causal concepts and grounded claims rather than prose similarity. Its 30-game validation split is grouped away from development and excluded from prompts and tuning. Full third-party annotations remain external; [`docs/commentary-quality.md`](docs/commentary-quality.md) defines the promotion roadmap.

Chess Scan is independently branded and does not redistribute workbook text, diagrams, answer keys, logos, or trade dress. Each review covers one corrected FEN and one observed learner attempt; engine continuations are always marked hypothetical. Opening history, player intent, and whole-game accuracy remain out of scope without PGN history. See [`docs/position-analysis.md`](docs/position-analysis.md) for the contract and held-out evaluation.

### Optional coaching selector

The deterministic review does not require an external service. To opt in, configure `CHESS_SCAN_COMMENTARY_PLANNER_ENABLED=true` together with an OpenAI-compatible endpoint and model using the variables in [`.env.example`](.env.example). The server sends only a bounded FEN/Stockfish/evidence packet, rejects any output beyond allowed claim IDs, and returns safe fixed copy on timeout or invalid output. An immutable attempt ledger and database leases provide cross-worker single-flight behavior and shared concurrency enforcement; dedicated executors plus per-confirmed-position and deployment-wide hourly budgets limit scanner pressure and external cost. Provider redirects are disabled and non-loopback endpoints require HTTPS. Ratings snapshot the exact coaching run actually shown. Provider selection and data-processing terms remain an operator decision.

### Stockfish licensing

Chess Scan remains MIT-licensed. `stockfish@18.0.8` is a separately distributed GPLv3 component: `web/scripts/copy-stockfish.mjs` copies its pinned lite single-thread Worker/WASM build into the generated `web/public/stockfish/` directory during development and production builds. The generated directory includes the complete GPLv3 license, source/build provenance, npm integrity, and SHA-256 hashes. It is excluded from Git because it is reproducibly sourced from the pinned package.

### Chess piece artwork licensing

The board pieces adapt [Chess Simple Assets](https://www.figma.com/community/file/971870797656870866/chess-simple-assets) by [Maciej Świerczek](https://www.figma.com/@swierq), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Chess Scan extracts and normalizes the SVGs, changes their palette and optical sizing, and redraws its annotation system in the same bold two-tone visual language. The original export, attribution, modification notice, and full license are retained under [`web/src/assets/chess-simple/`](web/src/assets/chess-simple/).

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

The active `chess-steps-v5` model uses the 2.6 MB v4 square CNN, then recovers a diagnosed regression on a real photographed Chess Steps workbook. Its external `platforms-v1` corpus contains 3,984 FEN-labelled boards spanning 83 visible piece styles, with the final 12 positions per style held out. The separately grouped `print-regressions-v1` corpus retains one consented rectified crop—not the full photograph—and is now an immutable promotion gate. Intentionally concealed themes—Chess.com Blindfold and Lichess Mono/Disguised—are excluded because their missing pixel information cannot be classified.

The labeled image corpora are deliberately outside Git. Prepare or sync the verified copies, then reproduce the model and run every gate:

```bash
make prepare-argus-data     # ~/chess-scan-training/argus-2026-03-29
make prepare-platform-data  # ~/chess-scan-training/platforms-v1 after acquiring assets
make prepare-lichess-puzzles # ~/chess-scan-training/lichess-puzzles-2026-07-05
make prepare-commentary      # data/qa-cache/expert-commentary-2026-07-22-v2
make prepare-commentary-shadow # data/qa-cache/expert-commentary-shadow-2026-07-22-v1
make train-platform-model
make train-print-recovery
make qa-argus
make qa-platform
make qa-print
make qa-review
make qa-commentary
make qa-coaching
make qa-online
make qa-stress
```

Set `CHESS_SCAN_ARGUS_DATA_DIR`, `CHESS_SCAN_PLATFORM_DATA_DIR`, `CHESS_SCAN_PRINT_DATA_DIR`, and `CHESS_SCAN_LICHESS_PUZZLE_DATA_DIR` to use other external locations. The manifests under [`benchmarks/`](benchmarks/) hash-describe the external corpora without redistributing source images or the 1,000-record puzzle splits. Production mounts the vision-training corpora read-only for automatic replay and gating; `s46-infra/hetzner/scripts/sync-chess-training-data.sh` explicitly pushes or pulls them with rsync.

V5 fixes the reference workbook from 62/64 to 64/64 squares and stays exact across eight fixed degradation variants. On the grouped clean platform holdout it improves v4 from 982 to 987 exact boards and from 63,724 to 63,729 correct squares. The deterministic camera/display gate retains 980/996 exact boards while improving by one square. Clean Chess.com is 454/456 exact, Lichess improves to 473/480, and Take Take Take remains 60/60. Two separate real Take Take Take app boards and the latest Chess.com Glass board remain exact.

Official source files are downloaded during training or QA and are not redistributed. V5 remains exact on all 267 reproducible online boards and every enforced print/photo gate, improves the Argus held-out sample from 4,495/4,500 to 4,496/4,500 without regressing any class, and improves synthetic replay from 19,378/19,500 to 19,382/19,500. The diagnosis and full gate comparison are recorded in [`docs/print-regression-v5.md`](docs/print-regression-v5.md).

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
uv run python scripts/export_review_feedback.py --rating unhelpful
uv run python scripts/adjudicate_review_feedback.py REVIEW_FEEDBACK_ID \
  --reviewer NAME --disposition approved_fix --notes REASON \
  --regression-fixture tests/test_review.py::TEST_NAME
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

`models/argus-v2r5.onnx` is the MIT-licensed Argus overlay square classifier. `models/chess-steps-v1.onnx` is the original print adaptation, `chess-steps-v2.onnx` is the queen-color revision, `chess-steps-v3.onnx` restores Argus coverage, `chess-steps-v4.onnx` adds multi-platform coverage, and `chess-steps-v5.onnx` restores real photographed-workbook coverage without regressing those domains. Trainable checkpoints and metadata are stored alongside the immutable weight artifacts.

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
