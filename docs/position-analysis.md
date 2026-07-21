# Position analysis

Chess Scan reviews one corrected FEN and one optional learner attempt. It does not infer game history, opening names, move intent, or whole-game accuracy.

## Pipeline

1. Stockfish 18 lite runs in a browser Worker and returns three candidate lines, root-side scores, WDL, depth, and a stable best move.
2. A learner attempt is force-analyzed from the same root. Every score uses the original side-to-move point of view.
3. The API validates every legal move and rejects bound, unstable, malformed, or ambiguously ranked analysis.
4. Deterministic detectors emit typed evidence: scope, proof, ply, actor, targets, causal move endpoints, squares, and legal moves.
5. The planner prefers a concrete refutation of a non-equivalent attempt, then a causal explanation of the best line, and otherwise abstains.
6. Human-authored text renders only selected evidence. Each annotation carries its line scope and ply so the board replays to the proven position before drawing an arrow. Candidate and refutation lines are labelled hypothetical, never played history.
7. The immutable request, response, engine attribution, schema version, and user rating are stored in SQLite.

The wire contract is versioned as `position-analysis-2`.

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

The gate also runs the production teaching planner. Legacy theme heuristics cannot contribute to that planner; all 1,150 mapped production claims on the held-out set come from the validated theme detectors and match Lichess metadata.

Promotion thresholds are 99.5% target and macro recall with setup evidence, 99.5% history-free accuracy for nonhistorical themes, and 99.5% mapped-claim precision in setup-aware, history-free, and production-planner modes.

## Feedback

Review runs, ratings, and expert adjudications are append-only. An unhelpful rating can include a fixed triage reason and optional detail. Feedback is evidence for expert review, not an automatic code or copy promotion signal. `scripts/adjudicate_review_feedback.py` records confirmed, rejected, duplicate, or approved-fix decisions; an approved fix is rejected unless it names a regression fixture. Detector changes must also pass the held-out gate.

Maia and model-generated prose are intentionally excluded. Stockfish supplies objective calculations; deterministic code and fixed copy provide the explanation.
