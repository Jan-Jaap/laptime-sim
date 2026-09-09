// Worker stop-semantics test: the annealing loop must observe shouldStop()
// between iterations and finish with exactly one result message.
//
// Run: node tests/js/worker.test.mjs   (part of `npm test`)

import assert from "node:assert";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { mulberry32 } from "../../src/streamlit_apps/components/raceline_optimizer/src/rng.ts";

const here = dirname(fileURLToPath(import.meta.url));
const fixture = JSON.parse(
  readFileSync(join(here, "fixtures", "build_batch.json"), "utf8").replace(/\bNaN\b/g, "null"),
);

const { runAnnealing } = await import(
  "../../src/streamlit_apps/components/raceline_optimizer/src/worker.ts"
);

const geometry = {
  width: Float64Array.from(fixture.geometry.width),
  pos_min: Float64Array.from(fixture.geometry.pos_min),
  pos_max: Float64Array.from(fixture.geometry.pos_max),
  is_circular: fixture.geometry.is_circular,
  idx_len: fixture.geometry.idx_len,
  n_coords: fixture.geometry.n_coords,
  left_coords: Float64Array.from(fixture.geometry.left_coords.flat()),
  right_coords: Float64Array.from(fixture.geometry.right_coords.flat()),
  slope: Float64Array.from(fixture.geometry.slope),
};
const state = {
  position: Float64Array.from(fixture.current_position),
  heatmap: new Float64Array(geometry.idx_len).fill(1),
  best_time: Math.min(...fixture.expected.laptimes.filter((t) => !Number.isNaN(t))),
  progress_rate: 1.0,
};

let stopped = false;
const messages = [];
await runAnnealing(state, geometry, fixture.car, {
  batchSize: 5,
  rng: mulberry32(7),
  progressIntervalMs: 1,
  shouldStop: () => stopped,
  onProgress: (msg) => {
    messages.push(msg);
    if (messages.length >= 2) stopped = true; // user pressed stop after two updates
  },
  onResult: (msg) => messages.push(msg),
});

const types = messages.map((m) => m.type);
assert(types.at(-1) === "result", `last message must be a result, got ${types.join(",")}`);
assert(types.filter((t) => t === "progress").length === 2, `expected 2 progress messages, got ${types.join(",")}`);
assert(Number.isFinite(state.best_time), "best_time stays finite");
console.log("worker stop semantics passed:", types.join(" -> "));
