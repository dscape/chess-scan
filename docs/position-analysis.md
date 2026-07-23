# Position analysis

Chess Scan reviews one corrected FEN and one optional learner attempt. It does not infer game history, opening names, move intent, or whole-game accuracy.

## Pipeline

1. Stockfish 18 lite runs in a browser Worker and returns three candidate lines, root-side scores, WDL, depth, and a stable best move.
2. A learner attempt is force-analyzed from the same root. Every score uses the original side-to-move point of view.
3. The API validates every legal move and rejects bound, unstable, malformed, or ambiguously ranked analysis.
4. Deterministic detectors emit typed evidence: scope, proof, ply, actor, targets, causal move endpoints, squares, and legal moves.
5. The planner prefers a concrete refutation of a non-equivalent attempt, then a causal explanation of the best line, and otherwise abstains.
6. Human-authored text renders only selected evidence. Each annotation carries its line scope and ply so the board replays to the proven position before drawing its diagram. Candidate and refutation lines are labelled hypothetical, never played history.
7. The API compiles semantic visual marks from that same evidence: played, engine, reply, attack, ray, and threat arrows; typed position markers; and annotation-level motif badges tied to a specific arrow. The frontend replays the declared geometry and suppresses redundant endpoint markers without inferring the tactic.
8. The deterministic review returns immediately. If an operator explicitly enables deeper coaching, the frontend separately asks a bounded model to select one or two already verified claims; the model cannot write claims, alter verdicts, or create diagrams.
9. Review and coaching runs, engine attribution, schema and prompt versions, raw planner output, accepted evidence IDs, latency, and user ratings are stored immutably in SQLite. Each rating snapshots the coaching state and exact run shown at submission time, so later planner completion cannot rewrite its context.

The wire contract is versioned as `position-analysis-5`. Version 5 adds explicit `counterfactual` proof alongside legal geometry and checked-line consequences, plus an authoritative concise attempt headline separate from detailed diagram annotations.

## Board diagrams

After an attempt, the board presents a short, selectable story rather than drawing every idea at once. A typical review shows the learner's move, the strongest hypothetical reply or causal motif, and the checked best-line idea. Every arrow role shares one shaft, head, and keyline geometry; color and dash pattern distinguish played moves, checked lines, tactical rays, and future threats. Fork diagrams branch from the landing square to their targets; pins and x-rays use tactical rays; future capture or mating threats use a dashed threat arrow. A compact motif token rides the relevant arrow so its meaning stays attached to the relationship it describes. Endpoint markers disappear when the arrow role already communicates their meaning; circular halos remain for clues and states such as danger, vacated, or blocked squares that a line alone does not express.

Every arrow, marker, and badge belongs to an annotation with one or more evidence IDs, and every motif badge names an arrow that crosses its evidence-backed anchor square. Its ply and move arrows must match the checked line, while relation arrows, markers, and badge anchors must match cited evidence geometry. Unsupported visuals are rejected, and quiet positions fall back to an engine comparison without inventing a tactical diagram.

## Attempt verdicts

Verdicts primarily use expected-score loss computed from Stockfish WDL, not exact UCI-move identity. Because WDL saturates in decisive positions, mate distance and centipawn deterioration break apparent ties before the WDL bands are applied. Giving up a forced mate is an inaccuracy even when the replacement line remains winning. This accepts sound alternatives without grading a much faster forced loss as good.

## Lichess theme gate

`make prepare-lichess-puzzles` checksum-verifies the official public-domain archive and creates deterministic, non-overlapping development and validation sets grouped by Lichess game ID. Each set has 1,000 puzzles from 1,000 distinct games: 100 for each supported theme.

Supported themes are attraction, capturing defender, clearance, discovered attack, fork, interference, intermezzo, pin, trapped piece, and x-ray attack.

`make qa-review` evaluates the held-out validation set. With the archive's observed setup move available to the evaluator, current frozen results are:

- target-theme accuracy: 99.8% (998/1,000);
- macro target-theme accuracy: 99.8%;
- mapped-claim precision: 99.77% (1,290/1,293).

A scanned FEN does not contain that setup move. The runtime therefore disables `intermezzo` teaching rather than implying a prior capture. Across the other nine metadata themes, history-free accuracy is 99.78% (898/900) with 100% mapped-claim precision (1,176/1,176). The two misses are `trappedPiece` records that the current public Lichess tagger no longer reproduces from the archived metadata. They remain in the gate rather than being silently relabelled.

The gate also runs the production teaching planner. Legacy theme heuristics cannot contribute to that planner; all 1,175 mapped production claims on the held-out set come from the validated theme detectors and match Lichess metadata. Strategic root evidence is selected before later-line motifs, while up to three validated tactical findings are retained behind it so tactical recall does not regress.

Promotion thresholds are 99.5% target and macro recall with setup evidence, 99.5% history-free accuracy for nonhistorical themes, and 99.5% mapped-claim precision in setup-aware, history-free, and production-planner modes.

## Expert commentary gate

`make qa-commentary` reconstructs reviews for 40 exact expert-annotated positions and compares the output with independently paraphrased claims. It scores move verdicts, supported first moves, primary concepts, later-line distractions, and explicitly disallowed primary lessons. It never compares prose wording.

The original ten Tata Steel 2026 positions remain development data because they were inspected while designing the contract. Version 2 adds a frozen validation split of 30 positions from 30 games across the 2025 Grand Swiss, 2023 World Championship, 2025 Women's World Championship, and 2025 World Cup studies. The validation sources do not overlap development and are never prompt or tuning inputs.

Full annotations and game scores remain external. The manifest stores collection and chapter URLs, source hashes, derived claims, game groups, and pinned Stockfish inputs. `make prepare-commentary` verifies every full study, selected chapter, exact FEN, move, and attached annotation hash. The original topic-only diagnostic reported 8 / 2 / 0 on development and 8 / 7 / 15 on validation after the strategic evidence pass. The promotion evaluator now also requires visible evidence to match the reference claim's causal move, proof, squares, pieces, and continuation; under that stricter contract the current output scores 6 / 1 / 3 and 6 / 0 / 24. A separately curated post-implementation 2024 World Championship shadow set scores 3 / 0 / 7 across ten new games. All three splits retain complete verdict and best-move agreement. See [`commentary-quality.md`](commentary-quality.md).

## Feedback

Review runs, ratings, and expert adjudications are append-only. An unhelpful rating can include a fixed triage reason and optional detail. Feedback is evidence for expert review, not an automatic code or copy promotion signal. `scripts/adjudicate_review_feedback.py` records confirmed, rejected, duplicate, or approved-fix decisions; an approved fix is rejected unless it names a regression fixture. Detector changes must also pass the held-out gate.

Maia and model-generated chess prose are intentionally excluded. Stockfish supplies objective calculations; deterministic code and fixed copy provide every factual explanation. The optional model only orders allowed claim IDs from a bounded packet and falls back to the deterministic review on any failure.
