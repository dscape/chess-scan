import test from "node:test";
import assert from "node:assert/strict";
import {
  cueAccessibleLabel,
  cueRoleDescription,
  cueRoleMark,
  displayedBoardCue,
  displayCueLabel,
  hasBoardCue,
} from "../src/reviewCue.ts";

const root = { id: "root", ply: 0 };
const later = { id: "later", ply: 2 };
const other = { id: "other", ply: 1 };

test("shares one visual-cue predicate for annotations", () => {
  const cue = { markers: [], arrows: [], badge: null };

  assert.equal(hasBoardCue(cue), false);
  assert.equal(hasBoardCue({ ...cue, arrows: [{}] }), true);
  assert.equal(hasBoardCue({ ...cue, markers: [{}] }), true);
  assert.equal(hasBoardCue({ ...cue, badge: {} }), true);
});

test("turns internal cue labels into concise board-story labels", () => {
  assert.equal(displayCueLabel("Reply · Winning material"), "Winning material");
  assert.equal(displayCueLabel("Your move"), "Your move");
  assert.equal(cueRoleMark({ arrows: [{ role: "reply" }] }), "!");
  assert.equal(cueRoleMark({ arrows: [{ role: "engine" }] }), "✦");
  assert.equal(
    cueRoleDescription({ arrows: [{ role: "reply" }] }),
    "Hypothetical reply",
  );
  assert.equal(cueRoleDescription({ arrows: [{ role: "engine" }] }), "Engine line");
  assert.equal(
    cueAccessibleLabel({
      label: "Reply · Winning material",
      text: "Nxd4 wins the knight.",
      arrows: [{ role: "reply" }],
    }),
    "Hypothetical reply: Winning material. Nxd4 wins the knight. Show on board.",
  );
});

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
