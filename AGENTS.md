# Chess Scan

Mobile-first Chess Steps diagram scanner. One photographed diagram is rectified, classified into 64 square labels, corrected by the user, and opened in Lichess.

## Commands

```bash
make install       # Python + web dependencies
make dev           # API on :8000 and Vite on :5173
make test          # backend tests
make check         # backend lint/tests and frontend typecheck/build
make build         # production web build
```

## Architecture

- `server/chess_scan/`: FastAPI, OpenCV geometry, ONNX inference, SQLite feedback store
- `web/`: React/Vite mobile interface
- `models/`: immutable ONNX artifacts and metadata
- `scripts/`: feedback export, candidate training, and model promotion
- `docs/learning-loop.md`: supervised and preference-learning process

## Invariants

- Runtime inference returns raw model predictions. Never silently repair king counts or other pieces.
- A confirmed board is an immutable human-feedback event tied to its model version.
- Training uses only explicitly confirmed records with training consent.
- Candidate models never become active without held-out evaluation gates.
- Store rectified board crops for training; delete temporary full photographs after confirmation.
- Production code must not import Argus or depend on its video, physical-board, or database pipelines.
