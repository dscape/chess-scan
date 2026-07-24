import assert from "node:assert/strict";
import test from "node:test";

import { ApiError, apiFailureFrom } from "../src/api.ts";

test("classifies retryable API and transport failures", () => {
  assert.deepEqual(
    apiFailureFrom(new ApiError("Busy", 503, null)),
    { kind: "api", message: "Busy", status: 503, retryable: true },
  );
  assert.deepEqual(
    apiFailureFrom(new TypeError("Network unavailable")),
    {
      kind: "api",
      message: "Network unavailable",
      status: null,
      retryable: true,
    },
  );
});

test("keeps permanent API failures distinct from local invariants", () => {
  assert.deepEqual(
    apiFailureFrom(new ApiError("Invalid analysis", 422, null)),
    {
      kind: "api",
      message: "Invalid analysis",
      status: 422,
      retryable: false,
    },
  );
  assert.equal(apiFailureFrom(new Error("Local invariant")), null);
});
