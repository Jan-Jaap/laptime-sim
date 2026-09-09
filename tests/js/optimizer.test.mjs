// Golden parity tests: replays fixtures generated from the Python
// implementation (uv run python tests/js/generate_golden.py) against the
// TypeScript port of the optimizer kernels.
//
// Run: node tests/js/optimizer.test.mjs   (or: npm test)

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  annealStep,
  applyOutcome,
  buildBatch,
  simulateLaptime,
} from "../../src/streamlit_apps/components/raceline_optimizer/src/optimizer.ts";
import { mulberry32 } from "../../src/streamlit_apps/components/raceline_optimizer/src/rng.ts";

const here = dirname(fileURLToPath(import.meta.url));
const fixturesDir = join(here, "fixtures");
// Python's json.dump emits a bare NaN for float NaN (invalid JSON) — map to null
// and convert back to NaN where numeric arrays are read.
const load = (name) =>
  JSON.parse(readFileSync(join(fixturesDir, name), "utf8").replace(/\bNaN\b/g, "null"), (key, value) =>
    value === null && key !== "" ? NaN : value,
  );

let failures = 0;
function check(name, ok, detail = "") {
  if (ok) {
    console.log(`  ok  ${name}`);
  } else {
    console.error(`FAIL  ${name}${detail ? `: ${detail}` : ""}`);
    failures++;
  }
}

const approx = (a, b, rel) => Math.abs(a - b) <= rel * Math.max(1, Math.abs(b));
function arraysClose(actual, expected, rel, label) {
  if (actual.length !== expected.length) return `${label}: length ${actual.length} != ${expected.length}`;
  for (let i = 0; i < expected.length; i++) {
    const e = expected[i];
    if (Number.isNaN(e)) {
      if (!Number.isNaN(actual[i])) return `${label}[${i}]: expected NaN, got ${actual[i]}`;
      continue;
    }
    if (!approx(actual[i], e, rel)) {
      return `${label}[${i}]: ${actual[i]} vs ${e} (diff ${Math.abs(actual[i] - e)})`;
    }
  }
  return null;
}

function parseGeometry(g) {
  return {
    width: Float64Array.from(g.width),
    pos_min: Float64Array.from(g.pos_min),
    pos_max: Float64Array.from(g.pos_max),
    is_circular: g.is_circular,
    idx_len: g.idx_len,
    n_coords: g.n_coords,
    left_coords: Float64Array.from(g.left_coords.flat()),
    right_coords: Float64Array.from(g.right_coords.flat()),
    slope: Float64Array.from(g.slope),
  };
}

// -- simulate: physics kernel ------------------------------------------------
{
  const f = load("simulate.json");
  const laptime = simulateLaptime(
    Float64Array.from(f.line_coordinates.flat()),
    Float64Array.from(f.slope),
    f.car,
  );
  check(
    "simulate laptime",
    approx(laptime, f.expected.laptime, 1e-9),
    `${laptime} vs ${f.expected.laptime}`,
  );
}

// -- build_batch: candidate construction (circuit + non-circular truncation) --
for (const name of ["build_batch.json", "build_batch_open.json"]) {
  const f = load(name);
  const result = buildBatch(
    parseGeometry(f.geometry),
    Float64Array.from(f.current_position),
    f.locations,
    f.lengths,
    f.deviations,
    f.car,
  );
  let detail = arraysClose(result.laptimes, f.expected.laptimes, 1e-9, "laptimes");
  check(`${name} laptimes`, detail === null, detail);
  for (let k = 0; k < f.expected.positions.length; k++) {
    detail = arraysClose(result.positions[k], f.expected.positions[k], 1e-12, `positions[${k}]`);
    check(`${name} positions[${k}]`, detail === null, detail);
    detail = arraysClose(result.perturbations[k], f.expected.perturbations[k], 1e-12, `perturbations[${k}]`);
    check(`${name} perturbations[${k}]`, detail === null, detail);
  }
}

// -- apply_outcome: acceptance / decay / all-NaN ------------------------------
for (const name of ["apply_outcome_accept.json", "apply_outcome_decay.json", "apply_outcome_all_nan.json"]) {
  const f = load(name);
  const state = {
    position: Float64Array.from(f.initial.position),
    heatmap: Float64Array.from(f.initial.heatmap),
    best_time: f.initial.best_time,
    progress_rate: f.initial.progress_rate,
  };
  const improved = applyOutcome(
    state,
    Float64Array.from(f.laptimes),
    f.positions.map((a) => Float64Array.from(a)),
    f.perturbations.map((a) => Float64Array.from(a)),
  );
  check(`${name} improved`, improved === f.expected.improved, `${improved} vs ${f.expected.improved}`);
  check(
    `${name} best_time`,
    approx(state.best_time, f.expected.best_time, 1e-12),
    `${state.best_time} vs ${f.expected.best_time}`,
  );
  check(
    `${name} progress_rate`,
    approx(state.progress_rate, f.expected.progress_rate, 1e-9),
    `${state.progress_rate} vs ${f.expected.progress_rate}`,
  );
  let detail = arraysClose(state.position, f.expected.position, 1e-12, "position");
  check(`${name} position`, detail === null, detail);
  detail = arraysClose(state.heatmap, f.expected.heatmap, 1e-9, "heatmap");
  check(`${name} heatmap`, detail === null, detail);
}

// -- annealStep: sanity (own PRNG, no numpy-parity required) -------------------
{
  const f = load("build_batch.json");
  const geometry = parseGeometry(f.geometry);
  // clamp the start into the clearance bounds: a fresh zeros line is out of
  // bounds by construction, and only candidates get clipped in annealStep
  const state = {
    position: Float64Array.from(f.current_position, (v, i) =>
      Math.min(Math.max(v, geometry.pos_min[i]), geometry.pos_max[i]),
    ),
    heatmap: new Float64Array(geometry.idx_len).fill(1),
    best_time: Math.min(...f.expected.laptimes.filter((t) => !Number.isNaN(t))),
    progress_rate: 1.0,
  };
  const rng = mulberry32(42);
  let inBounds = true;
  for (let step = 0; step < 50; step++) {
    annealStep(state, geometry, f.car, 10, rng);
    for (let i = 0; i < geometry.idx_len; i++) {
      if (state.position[i] < geometry.pos_min[i] - 1e-12 || state.position[i] > geometry.pos_max[i] + 1e-12) {
        inBounds = false;
      }
    }
  }
  check("annealStep 50 steps stay within border clearance", inBounds);
  check(
    "annealStep best_time finite and not worse than start",
    Number.isFinite(state.best_time) && state.best_time <= f.expected.laptimes.reduce((a, b) => (b < a ? b : a)),
    `${state.best_time}`,
  );
}

if (failures > 0) {
  console.error(`\n${failures} parity check(s) failed`);
  process.exit(1);
}
console.log("\nall parity checks passed");
