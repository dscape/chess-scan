# Chess Steps benchmarks

These manifests reference official sample PDFs without redistributing their images.

- `chess-steps-step2.json`: 12 manually audited full FENs used as glyph references. The PDF pixels are excluded from training.
- `chess-steps-kings.json`: independent manually audited king locations from a different PDF. The entire PDF is excluded from template construction, training, and checkpoint selection.
- `chess-steps-online-sources.json`: URL, hash, and byte-size inventory for 184 official QA source artifacts plus the 12 shared interactive piece assets. It contains no source imagery.
- `chess-steps-german-manuals.json`: 63 standard diagram FENs from six official manual samples. Labels come from the audited glyph templates; the sole disagreement was visually adjudicated.
- `chess-steps-queen-colors.json`: source locators, augmentation settings, and labels for the reproducible queen-color adaptation and king retention set. Its end-to-end reconstruction is gate-equivalent to the promoted artifact; MPS output is not byte-deterministic.
- `qa-2026-07-18.json`: machine-readable corpus counts, metrics, stress results, and all manually adjudicated model/template disagreements.

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
```

Recreate the queen-color adaptation candidate:

```bash
uv run --extra ml --with 'pymupdf>=1.25,<2' \
  python scripts/train_queen_adaptation.py
```
