# Chess Steps benchmarks

These manifests reference official sample PDFs without redistributing their images.

- `chess-steps-step2.json`: 12 manually audited full FENs used as glyph references. The PDF pixels are excluded from training.
- `chess-steps-kings.json`: independent manually audited king locations from a different PDF. The entire PDF is excluded from template construction, training, and checkpoint selection.
- `chess-steps-online-sources.json`: URL, hash, and byte-size inventory for 184 official QA source artifacts plus the 12 shared interactive piece assets. It contains no source imagery.
- `chess-steps-german-manuals.json`: 63 standard diagram FENs from six official manual samples. Labels come from the audited glyph templates; the sole disagreement was visually adjudicated.
- `chess-steps-queen-colors.json`: source locators, augmentation settings, and labels for the reproducible queen-color adaptation and king retention set. Its end-to-end reconstruction is gate-equivalent to the promoted artifact; MPS output is not byte-deterministic.
- `qa-2026-07-18.json`: machine-readable corpus counts, metrics, stress results, and all manually adjudicated model/template disagreements.
- `argus-training-corpus.json`: external Argus archive, prepared-split, and replay hashes without source images.
- `platform-training-corpus.json`: external Chess.com, Lichess, and Take Take Take source and label-manifest hashes without redistributing platform artwork.
- `print-regression-corpus.json`: hash inventory for consented rectified workbook crops; full photographs and workbook pixels are not committed.
- `lichess-puzzle-corpus.json`: checksum, deterministic game-grouped sampling rules, and balanced split metadata for two external 1,000-puzzle theme benchmarks sourced from Lichess's public-domain puzzle database. The evaluator reports setup-aware, history-free, and production-planner agreement separately.
- `expert-commentary-v1.json`: archived original ten-position Tata Steel snapshot. It predates collection-level split metadata and is retained for audit only; the preparation and evaluation CLIs intentionally require v2 or later.
- `expert-commentary-v2.json`: those ten development cases plus 30 frozen validation games from four independent expert-annotated studies.
- `expert-commentary-shadow-v1.json`: ten post-implementation shadow games from the independently annotated 2024 World Championship study. It was selected after the deterministic and planner work and was not used to tune that candidate.

The commentary manifests commit derived concepts and typed claims with pinned Stockfish inputs, not article or annotation prose. Promotion credit requires a user-visible finding whose causal move, evidence kind, proof, squares, pieces, and continuation support one of those claims; hidden findings and topic-name matches do not count. Full studies and chapter PGNs remain external and checksum-verified.

Square indices are image-order values from `0` at the top-left to `63` at the bottom-right. Source hashes make an upstream sample change fail loudly instead of silently changing a benchmark.

Run the reproducible adaptation job from the repository root:

```bash
uv run --extra ml --with 'pymupdf>=1.25,<2' \
  python scripts/train_chess_steps_model.py
```

Rerun the independent online and deterministic photo QA gates:

```bash
make qa-online
make qa-stress
make qa-argus
make qa-platform
make qa-print
make qa-review
make qa-commentary
make qa-coaching
```

Recreate the disjoint Lichess development and validation splits from the checksum-pinned archive, or verify the external expert-commentary chapters:

```bash
make prepare-lichess-puzzles
make prepare-commentary
make prepare-commentary-shadow
```

Recreate the photographed-print recovery candidate after mounting all verified external corpora:

```bash
make train-print-recovery
```

Recreate the queen-color adaptation candidate:

```bash
uv run --extra ml --with 'pymupdf>=1.25,<2' \
  python scripts/train_queen_adaptation.py
```
