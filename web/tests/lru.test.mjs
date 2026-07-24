import assert from "node:assert/strict";
import test from "node:test";

import { readLru, storeLru } from "../src/lru.ts";

test("promotes a present undefined value", () => {
  const cache = new Map([
    ["first", 1],
    ["present", undefined],
    ["last", 3],
  ]);

  assert.equal(readLru(cache, "present"), undefined);
  assert.deepEqual([...cache.keys()], ["first", "last", "present"]);
});

test("evicts an undefined key without exceeding the limit", () => {
  const cache = new Map([[undefined, "oldest"]]);

  storeLru(cache, "new", "newest", 1);

  assert.equal(cache.size, 1);
  assert.deepEqual([...cache.entries()], [["new", "newest"]]);
});
