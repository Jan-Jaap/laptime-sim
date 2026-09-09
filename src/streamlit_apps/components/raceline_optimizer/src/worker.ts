// Web Worker: runs the annealing loop on the visitor's CPU, posting throttled
// progress and a final result. The claimed best_time is never trusted — the
// server re-simulates the posted position (registry.accept_result).
//
// The loop is async and yields to the event loop between iterations: the stop
// message is processed while the loop runs. A fully synchronous loop would
// block the worker thread and make stop impossible.

import {
  annealStep,
  simulateLaptime,
  type AnnealState,
  type CarParams,
  type TrackGeometry,
} from "./optimizer.ts";
import { mulberry32 } from "./rng.ts";

// mirrors the CLI convergence tolerance (src/laptime_sim/cli.py)
const TOLERANCE = 0.1;

export interface SerializedGeometry {
  width: number[];
  pos_min: number[];
  pos_max: number[];
  is_circular: boolean;
  idx_len: number;
  n_coords: number;
  left_coords: number[][]; // [n_coords][3]
  right_coords: number[][];
  slope: number[];
}

export interface WorkerStart {
  type: "start";
  geometry: SerializedGeometry;
  car: CarParams;
  position: number[];
  best_time: number; // server-simulated baseline; may be absent for a fresh line
  batch_size: number;
  seed: number;
  progress_interval_ms: number;
}

export interface AnnealRunOptions {
  batchSize: number;
  rng: () => number;
  progressIntervalMs: number;
  shouldStop: () => boolean;
  onProgress: (msg: Record<string, unknown>) => void;
  onResult: (msg: Record<string, unknown>) => void;
}

/** Yields to the event loop without setTimeout's clamping. */
function yieldToEventLoop(): Promise<void> {
  return new Promise((resolve) => {
    const channel = new MessageChannel();
    channel.port1.onmessage = () => {
      channel.port1.close();
      resolve();
    };
    channel.port2.postMessage(null);
  });
}

export async function runAnnealing(
  state: AnnealState,
  geometry: TrackGeometry,
  car: CarParams,
  opts: AnnealRunOptions,
): Promise<void> {
  const started = performance.now();
  let candidates = 0;
  let lastProgress = 0.0;

  for (let step = 0; !opts.shouldStop(); step++) {
    annealStep(state, geometry, car, opts.batchSize, opts.rng);
    candidates += opts.batchSize;
    const now = performance.now();
    if (now - lastProgress >= opts.progressIntervalMs) {
      lastProgress = now;
      const elapsed = (now - started) / 1000;
      opts.onProgress({
        type: "progress",
        best_time: state.best_time,
        candidates,
        rate: candidates / elapsed,
        progress_rate: state.progress_rate,
        finished: false,
      });
    }
    if (state.progress_rate < TOLERANCE) {
      break;
    }
    await yieldToEventLoop();
  }

  opts.onResult({
    type: "result",
    position: Array.from(state.position),
    best_time: state.best_time,
    candidates,
    finished: true,
  });
}

function toGeometry(g: SerializedGeometry): TrackGeometry {
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

function coordsFromPosition(geometry: TrackGeometry, position: Float64Array): Float64Array {
  const coords = new Float64Array(geometry.n_coords * 3);
  for (let i = 0; i < geometry.idx_len; i++) {
    for (let j = 0; j < 3; j++) {
      coords[i * 3 + j] =
        geometry.left_coords[i * 3 + j] +
        (geometry.right_coords[i * 3 + j] - geometry.left_coords[i * 3 + j]) * position[i];
    }
  }
  if (geometry.is_circular) {
    for (let j = 0; j < 3; j++) {
      coords[(geometry.n_coords - 1) * 3 + j] = coords[j];
    }
  }
  return coords;
}

let stopped = false;

async function startRun(msg: WorkerStart): Promise<void> {
  const geometry = toGeometry(msg.geometry);
  const car: CarParams = msg.car;
  const state: AnnealState = {
    position: Float64Array.from(msg.position),
    heatmap: new Float64Array(geometry.idx_len).fill(1),
    best_time: msg.best_time,
    progress_rate: 1.0,
  };
  if (!Number.isFinite(state.best_time)) {
    // no stored baseline: measure the starting line client-side
    state.best_time = simulateLaptime(coordsFromPosition(geometry, state.position), geometry.slope, car);
  }

  await runAnnealing(state, geometry, car, {
    batchSize: msg.batch_size,
    rng: mulberry32(msg.seed),
    progressIntervalMs: msg.progress_interval_ms,
    shouldStop: () => stopped,
    onProgress: (m) => postMessage(m),
    onResult: (m) => postMessage(m),
  });
}

// `self` only exists inside a real Web Worker; the guard keeps worker.ts
// importable in node (tests import runAnnealing directly).
if (typeof self !== "undefined") {
  self.onmessage = (event: MessageEvent) => {
    const msg = event.data as { type: string };
    if (msg.type === "stop") {
      stopped = true;
    } else if (msg.type === "start") {
      stopped = false;
      void startRun(msg as WorkerStart);
    }
  };
}
