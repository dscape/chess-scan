# Expert commentary quality loop

Chess Scan should agree with expert analysis about the useful reason for a move, not imitate an annotator's wording. Stockfish remains the authority for move quality. Explanations may use only legal geometry, checked engine lines, or explicitly labelled counterfactual analysis.

## Delivery plan

- [x] **1. Reproducible development benchmark**
  - Record the ten researched Tata Steel positions as derived, structured references.
  - Keep full third-party annotations outside Git; retain source URLs and hashes.
  - Pin the Stockfish request used for each comparison.
  - Add source verification, an evaluator, tests, and `make qa-commentary`.
- [x] **2. Held-out expert validation**
  - Add positions from different articles, events, games, and annotators.
  - Group splits by game and source; never use validation records in prompts or tuning.
  - Mark claims that require history or player intent as out of scope.
  - Freeze 30 positions from 30 games across four checksum-pinned studies before changing review rules.
- [x] **3. Deterministic review baseline**
  - Fix copy and equivalence defects exposed by the development set.
  - Prefer a causal root-move lesson over an unrelated motif later in the PV.
  - Add reusable evidence atoms for strategic plans without weakening tactical gates.
  - Cover temporary sacrifices, prophylaxis, pawn breaks, space, and supported pawn advances with legal or checked-line evidence.
- [x] **4. Evidence-constrained LLM planner**
  - Keep scores and verdicts deterministic.
  - Send only FEN, checked lines, and bounded evidence atoms to an opt-in planner.
  - Require typed claims with evidence IDs; reject unsupported output and fall back safely.
  - Store provider, model, prompt, evidence, raw output, accepted claims, token usage, and latency.
- [x] **5. Product integration**
  - Present the verified coaching layer without delaying the local Stockfish result.
  - Explain when deeper coaching is unavailable; never expose raw model failure text.
  - Preserve evidence-backed diagrams, accessibility, mobile layout, and feedback capture.
- [ ] **6. Promotion and continuous improvement**
  - [ ] Run repeated live generations because model output is nondeterministic.
  - [ ] Require hard factual validity, held-out semantic improvement, and blind human preference.
  - [x] Turn only expert-confirmed production problems into append-only regressions.
  - [x] Treat every prompt or provider-model revision as a new candidate.
- [ ] **7. Final verification**
  - [ ] Run backend, frontend, build, benchmark, and browser checks after the final edits.
  - [ ] Run Pi `/review` over the completed change, fix every valid finding, and repeat until clean.

## Reference contract

Each benchmark record separates:

- a move-quality verdict;
- acceptable primary and secondary concepts;
- FEN-provable and line-provable claims;
- disallowed primary explanations;
- history-dependent claims that are not scored;
- source and adjudication provenance.

The evaluator compares typed concepts and claims. It does not use text similarity, BLEU, embeddings, or an LLM judge as a promotion authority.

## Frozen baselines

The initial ten-case development baseline was 3 strong / 3 partial / 4 misses. The v2 validation split was frozen before deterministic or generated-commentary changes at 6 strong / 5 partial / 19 misses. All 30 move verdicts and engine choices agree with the pinned Stockfish input. The cases span 30 games, four source collections, and six annotators; no validation game or source collection overlaps development.

The deterministic evidence pass originally reported 8 strong / 2 partial / 0 misses on development and 8 strong / 7 partial / 15 misses on validation. A subsequent review found that those figures credited hidden findings and topic-name matches even when the visible evidence proved a different move or claim. Those concept-only figures are retained as historical diagnostics, not promotion evidence.

The corrected evaluator requires a user-visible finding whose causal move, evidence kind, proof, squares, pieces, and checked continuation match a typed reference claim. Under that stricter contract, the candidate scores 6 strong / 1 partial / 3 misses on development and 6 strong / 0 partial / 24 misses on validation. Verdict and best-move support remain 100%. The validation score rose by one after the temporary-sacrifice detector began requiring the root move to cause the capture offer; this removed an unrelated false primary lesson and exposed the already-supported pawn-break explanation. The frozen release floor remains unchanged pending human promotion review.

A post-implementation shadow set was frozen from ten different games in the independently annotated 2024 World Championship study. Without tuning against it, the evidence-qualified result is 3 strong / 0 partial / 7 misses, with 100% verdict and best-move support, one disallowed primary, and no later-line primary. For example, a hidden space finding no longer rescues a visible disallowed clearance lesson, and `...h5` no longer earns credit for an expert claim about `...g5`. `make qa-commentary` runs development, validation, and shadow gates.

Validation and shadow records must never be copied into prompts or used to choose rules. Any later rule or prompt candidate needs another fresh shadow adjudication.

## Development-only editorial rubric

The development corpus contributes structure, never reusable sentences. A useful coaching note follows four independently written moves:

1. **Diagnose the observable cause.** Name the move and the verified consequence. Do not infer intent, preparation, psychology, or whether the player calculated.
2. **Show the checked branch.** Put the forcing reply before general explanation and cite enough ordered legal plies to reach the claimed idea.
3. **Compare honestly.** Call Stockfish's first choice “better” only when the attempt is not equivalent. For equivalent moves, describe another sound approach; for the same first move, avoid a redundant comparison.
4. **Leave a reusable calculation cue.** Turn the position into an action such as checks first, captures first, or testing the opponent's strongest reply. State what to inspect, not what the player supposedly failed to think about.

Use concise cause-and-effect prose, preserve SAN capture/check/mate/promotion notation, and let interactive move tokens carry line citations. Omit internal scope names, claim numbers, provider state, and fallback jargon. Never copy article prose or expose development examples to validation/shadow prompts.

## Planner contract

The optional `commentary-planner-2` integration keeps the model as a bounded selector rather than a chess analyst. The deterministic review is returned first. A separate coaching request sends an OpenAI-compatible endpoint only the FEN, checked Stockfish lines, deterministic evidence atoms, and pre-rendered claim candidates. It never sends a photograph, scan or feedback ID, user text, article annotation, or training record.

The `commentary-selection-2` prompt may return only one or two allowed claim IDs plus a fixed focus enum from a `commentary-evidence-2` packet. When a non-equivalent attempt has a checked refutation, the packet names that claim as the required primary lesson; a response that puts another idea first is rejected. The focus deterministically reorders cause, concept, or comparison sections, and positions whose presentation cannot change skip the provider entirely.

The server turns selected verified facts into an editorial coaching note: a concrete diagnosis, a checked continuation, a better option, and an observable calculation habit. Every displayed SAN move is a structured reference to one ply of the canonical best or attempt line; canonical moves embedded in evidence copy become the same interactive tokens, while a noncanonical counterfactual is explicitly described as counterfactual prose. The client replays exact cited positions on the board. Development-source analysis informed this cause → line → comparison → practice structure, but source prose is never copied and validation/shadow records never enter prompts.

Unknown IDs, extra fields, duplicate JSON keys, malformed JSON, invalid provider envelopes, bounded end-to-end timeouts, oversized responses, and provider failures use the same deterministic narrative ordering. Candidate-free and unavailable coaching do not expose internal “fallback” labels in the interface. Unexpected internal planner defects are logged and are not cached as final runs. Scores, verdicts, moves, diagrams, and factual wording cannot be changed by the model. One immutable run is retained per review with provider/model/prompt versions, the exact sanitized provider request, raw output, accepted IDs, rendered narrative, latency, token usage, and a sanitized failure code. Stored accepted output is rechecked with the same bounded duplicate-key-rejecting parser, while legacy V1 rows retain their exact stored snapshot and receive a version-aware V2 serving projection. The API key is never persisted.

An immutable admission ledger counts every potentially paid attempt even after a worker failure. A unique per-review attempt plus fenced database leases prevents duplicate paid calls and enforces provider concurrency across workers. Timed-out transport reservations remain active until the underlying transport future completes (with a crash-recovery lease as a backstop). Dedicated bounded preflight, coaching, and HTTP executors keep slow external I/O and burst queues out of the scanner pool; configurable per-confirmed-position and deployment-wide hourly budgets cap spend even if review or confirmation IDs are created repeatedly. `provider_called` separately records whether transport actually started. Candidate-free and pre-transport fallbacks record `provider_called=false`; candidate-free fallbacks consume no provider budget. Ratings snapshot whether coaching was not shown, loading, unavailable, disabled, accepted, or fallback; accepted/fallback ratings reference the exact immutable planner run rather than attaching a later result during export.

The provider is disabled by default. Operators must explicitly configure `CHESS_SCAN_COMMENTARY_PLANNER_*`, choose an endpoint/model compatible with their privacy policy, and accept that the bounded position packet leaves the server. Provider redirects are disabled; non-loopback endpoints must use HTTPS; endpoint URLs containing user-info, query parameters, or fragments are rejected so credentials cannot enter persisted request snapshots. `make qa-coaching` adversarially checks accepted selections and fallback behavior without an external call. Live promotion additionally requires the first selected lesson to be no worse than deterministic order and to choose the best evidence-qualified candidate available in every scorable case. A release candidate must run repeated live generations with explicit cost consent:

```bash
uv run python scripts/evaluate_commentary_planner.py \
  --split validation --repetitions 3 --live --confirm-provider-cost
```

## Promotion principles

Hard failures always block user-visible generated coaching: illegal moves, wrong pieces or squares, unsupported claims, scope errors, invented history, or a changed engine verdict. Subjective quality is evaluated separately by concept coverage and blind human comparison. A failed or slow planner returns the deterministic review.
