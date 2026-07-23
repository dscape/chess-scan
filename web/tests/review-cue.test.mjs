import test from "node:test";
import assert from "node:assert/strict";
import { displayedBoardCue } from "../src/reviewCue.ts";

const root = { id: "root", ply: 0 };
const later = { id: "later", ply: 2 };
const other = { id: "other", ply: 1 };

test("preserves an automatic later-ply cue while that annotation is hovered", () => {
  assert.equal(displayedBoardCue(null, later, later, root, 2), later);
});

test("suppresses an automatic cue while a different annotation is hovered", () => {
  assert.equal(displayedBoardCue(null, other, later, root, 2), null);
});

test("keeps pin and root-cue precedence", () => {
  assert.equal(displayedBoardCue(other, root, later, root, 0), other);
  assert.equal(displayedBoardCue(null, root, later, root, 0), root);
});
