# QA report: online Chess Steps coverage and mobile usability

Date: 2026-07-18  
Runtime and weight revision: `chess-steps-v2`  
Artifact: `51049359e9b9e4f2b101d2f762adc2ca049dc48ed515e9b9e6778d1efd12a363`

## Scope

This pass tested unseen official material, photo degradation, full geometry-to-FEN behavior, and the 390×844 mobile interface. It hash-verified 184 additional official source artifacts and deliberately does not count the 24 English PDFs used for the original adaptation as an independent holdout.

No official source image is stored in the repository. The QA corpus was downloaded temporarily from:

- Dutch samples: <https://www.stappenmethode.nl/nl/download.php>
- German samples: <https://www.stappenmethode.nl/de/download.php>
- French samples: <https://www.stappenmethode.nl/fr/download.php>
- Interactive workbook examples: <https://www.stappenmethode.nl/en/example-lessons.php>
- German manual samples: <https://www.stappenmethode.de/leseproben>
- Stepping Stones samples: <https://www.chess-steps.com/book-stepping-stones>
- Step 6 correction sheet: <https://www.chess-steps.com/Images/step6_3.pdf>

## Corpus

| Slice | Count | Treatment |
| --- | ---: | --- |
| Additional localized official PDFs | 70 | Rendered at 3× and kept out of training |
| Candidate diagrams extracted from localized PDFs | 3,293 | Deduplicated by normalized board-pixel hash |
| Unique localized images not exactly present in English adaptation data | 1,078 | Primary printed-diagram holdout |
| Localized standard positions with exactly one king of each color | 976 | Template-labeled, then every model/template disagreement manually adjudicated |
| Localized holdout images whose 64-label placement was absent from adaptation data | 358 | Novel-position subset |
| Official interactive exercise sets | 17 | 204 unique positions with source-embedded FEN labels |
| Interactive positions absent from adaptation data | 191 | Direct-label novel-position subset |
| Reproducible German manual sample PDFs | 6 | 67 diagrams extracted; 63 standard positions |
| German manual positions absent from adaptation data | 63 | All distinct from the localized and interactive position sets |
| Additional instructional PDFs | 73 | 2,634 diagrams; 1,632 unique after prior-corpus deduplication |
| Additional two-king reminder/manual diagrams | 1,376 | Broad glyph/notation stress slice |
| Nonstandard grids with exercise marks or missing kings | 85 | Reported separately; not treated as ordinary FEN positions |
| Ambiguous or non-grid extractions rejected from those slices | 17 | Excluded by checkerboard-grid scoring |
| Stepping Stones diagrams from 17 official web images | 64 | Resolution/geometry stress only; each source board is roughly 100 px wide |

Translated copies were not allowed to inflate the result: 1,944 localized board images were exact duplicates of English adaptation images and were removed. The template labeler had already scored 99.7% against the manually audited Step 2 page. A second audit of all 54 disputed squares across 49 standard holdout boards found that the base model was correct on 45 and the template was correct on nine. Those nine were genuine white-queen-as-black-queen errors missed in the first adjudication. `chess-steps-v2` fixes all nine without changing any other clean localized prediction.

## Accuracy

### Unseen standard diagrams

| Metric | Result |
| --- | ---: |
| Localized exact king locations | 976 / 976 |
| Base-model board-exact after corrected adjudication | 967 / 976 |
| `chess-steps-v2` board-exact after corrected adjudication | 976 / 976 |
| Localized novel-position king locations | 358 / 358 |
| Raw base-model/template board agreement before adjudication | 927 / 976 |
| Raw localized square agreement before adjudication | 99.91% |
| Interactive source-FEN board exact | 204 / 204 |
| Interactive source-FEN square exact | 13,056 / 13,056 |
| Interactive exact king locations | 204 / 204 |
| German manual board-exact after disagreement adjudication | 63 / 63 |
| German manual exact king locations | 63 / 63 |
| Combined standard source images board-exact | 1,243 / 1,243 |
| Unique tested piece placements | 1,022 |
| Unique placements absent from adaptation data | 597 |

The interactive examples use the FEN embedded in the official exercise JavaScript and the official 40 px piece assets, so this slice does not depend on template labels. The German manual slice uses the audited glyph templates; its sole model/template disagreement was visually confirmed as a white queen correctly read by the model. The model retained the prior independent results: 12/12 manually audited king positions, 12/12 manually audited full boards, and 1,064/1,064 exact positions in the English official corpus after adjudication.

### Expanded instructional PDFs

The broader 73-PDF crawl produced 1,376 unique new diagrams with one template king of each color. Inspection of every disagreement excluded ten instructional-symbol or malformed extraction cases. On the remaining 1,366 diagrams:

| Metric | Result |
| --- | ---: |
| Exact king locations | 1,366 / 1,366 |
| Board-exact after adjudication | 1,364 / 1,366 |
| Square accuracy after adjudication | 99.998% |
| Remaining errors | 2 white queens read as black queens |

Both remaining queens trigger the new per-square review warning. More aggressive queen fine-tuning fixed them but changed already-correct clean diagrams, so that checkpoint was rejected. The promoted `chess-steps-v2` checkpoint is the strongest no-clean-regression candidate.

### Photo and print stress

The 12 manually audited Step 2 boards were transformed with random perspective, page-colored backgrounds, blur, JPEG loss, reduced contrast, and downsampling. These tests pass through board detection, rectification, and ONNX inference.

| Test | Before QA fixes | After QA fixes |
| --- | ---: | ---: |
| Moderate perspective: boards detected | 5 / 12 | 12 / 12 |
| Moderate perspective: board-exact | 5 / 12 | 12 / 12 |
| Severe perspective + blur: boards detected | 5 / 12 | 12 / 12 |
| Severe perspective + blur: board-exact | 5 / 12 | 12 / 12 |
| Faded print, board ≥256 px: board-exact | 0 / 12 | 12 / 12 |
| Clean downsample, board 256 px: board-exact | 12 / 12 | 12 / 12 |
| Board 128 px or smaller | Unreliable | Explicitly treated as below the quality floor |

Fitting the outer board to all 49 detected internal intersections, rather than only the four extreme intersections, eliminated the remaining edge-square rectification failure.

### Nonstandard exercise diagrams

Some early and Plus exercises draw `+`, `?`, circles, route dots, or other teaching marks in squares. These are not chess pieces and do not have a unique FEN interpretation.

On 85 unseen marked grids, the deployed model often interpreted marks as kings. A targeted fine-tuning experiment reduced false symbol-kings from 86 to 7, but it also changed genuine rooks, queens, and one king on the clean holdout. The candidate was rejected rather than trading normal-board correctness for notation handling.

The product now surfaces king-count problems near the top of the review, so marked exercises require explicit human correction. They are not included in the standard-board accuracy claim.

### Stepping Stones web previews

The official Stepping Stones web images are 300×400 page thumbnails containing six boards. Individual boards are only about 100×100 px and the photographed pages are curved. Automatic geometry succeeded on 15/64 after enhancement, which is below a usable threshold. These files also violate the product capture contract: one board should fill the camera guide.

This is a source-resolution limit, not a claim about a close phone photograph of a Stepping Stones board. The camera now offers manual capture when automatic locking cannot complete, followed by four-corner adjustment. Low-resolution scans are clearly warned instead of presented as reliable.

## Usability review

Tested in Chromium at 390×844 with mocked live camera streams for an undetectable frame and a faded perspective board.

| Before | After |
| --- | --- |
| Camera could wait indefinitely when a faint board never locked | Added a visible `Capture now` shutter that proceeds to four-corner adjustment |
| Existing-photo fallback was available only before camera entry or after permission failure | Added an always-available `Photos` action in the live camera |
| Low-contrast preview detection failed most perspective stress cases | Added CLAHE-assisted checkerboard detection; all 12 stress boards now lock |
| Outer-corner extrapolation trusted only four noisy intersections | Fit the homography across all 49 internal intersections; both perspective sets are now 12/12 board-exact |
| Faded diagrams reached a model trained on normal contrast | Added guarded luminance normalization for sharp, faded board crops |
| Very small boards could look authoritative after upscaling | Added a prominent warning below a 256 px detected board edge |
| One ambiguous piece could be hidden by a high 64-square average | Added per-square warning counts, including explicit white/black queen ambiguity |
| Manual framing uncertainty was communicated only in the collapsible frame panel | Added a top-level quality banner and automatically opened corner adjustment |
| Extra or missing kings appeared only as chips near the final FEN | Added king-count warnings to the top-level quality banner |
| No clear recovery from failed automatic capture | Manual shutter successfully transitions to the corner editor and re-read action |

The camera and editor had no horizontal overflow at 390 px. Interactive camera controls meet the 40 px minimum target. The piece picker exposes all 13 labels with accessible names, remains within the viewport, and the faded-camera flow produced the exact expected FEN.

## Reproduction and audit trail

The 184-source URL/hash inventory is [`benchmarks/chess-steps-online-sources.json`](../benchmarks/chess-steps-online-sources.json). German manual labels are in [`benchmarks/chess-steps-german-manuals.json`](../benchmarks/chess-steps-german-manuals.json), and the queen-color adaptation locators are in [`benchmarks/chess-steps-queen-colors.json`](../benchmarks/chess-steps-queen-colors.json). Machine-readable metrics and adjudications are in [`benchmarks/qa-2026-07-18.json`](../benchmarks/qa-2026-07-18.json). None of these files contains source imagery.

Run the independently labeled online and deterministic photo gates with:

```bash
make qa-online
make qa-stress
```

`qa-online` redownloads and hash-verifies 17 official interactive position sets, their piece assets, and six German manual PDFs: 267/267 reproducible source-labeled or adjudicated boards. `qa-stress` redownloads and hash-verifies the manually audited Step 2 PDF, renders its reference page, and reruns the fixed transformations. Both commands cache downloads under the gitignored `data/qa-cache` directory.

Recreate the queen-color candidate from source locators and Argus replay with:

```bash
uv run --extra ml --with 'pymupdf>=1.25,<2' \
  python scripts/train_queen_adaptation.py
```

The reconstruction was executed end-to-end from all 167 hash-verified PDFs and 4,893 source locators. MPS training was not byte-identical, but the reproduced artifact matched every promotion result: 976/976 localized boards after adjudication, 1,364/1,366 expanded boards, 1,366/1,366 expanded kings, 19,306/19,456 Argus replay squares, 267/267 online boards, and every 12-board photo gate.

## Current acceptance boundary

Reliable:

- one standard Chess Steps board filling the guide
- official English, Dutch, German, and French workbook glyphs
- moderate perspective and faded print when the board is at least 256 px across
- manual correction, orientation, side-to-move, and Lichess handoff

Requires human correction:

- exercise notation drawn inside squares
- boards without exactly one king of each color
- source boards below roughly 256 px
- severely curved pages where a homography cannot make all square boundaries uniform
