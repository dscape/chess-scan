import assert from "node:assert/strict";
import test from "node:test";

import {
  loadDataSharingConsent,
  saveDataSharingConsent,
} from "../src/dataSharing.ts";

test("keeps scanner data sharing off until yes is persisted", () => {
  const values = new Map();
  const storage = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
  };

  assert.equal(loadDataSharingConsent(storage), false);
  saveDataSharingConsent(storage, true);
  assert.equal(loadDataSharingConsent(storage), true);
  saveDataSharingConsent(storage, false);
  assert.equal(loadDataSharingConsent(storage), false);
});

test("keeps sharing off when browser storage is unavailable", () => {
  const blocked = {
    getItem: () => { throw new Error("blocked"); },
    setItem: () => { throw new Error("blocked"); },
  };

  assert.equal(loadDataSharingConsent(blocked), false);
  assert.doesNotThrow(() => saveDataSharingConsent(blocked, true));
  assert.equal(loadDataSharingConsent(null), false);
});
