// TypeScript port of the laptime-sim hot kernels:
//   simulate_kernel / calc_speed  (src/laptime_sim/simulate.py)
//   _batch_kernel                 (src/laptime_sim/optimizer.py)
//   _apply_outcome                (src/laptime_sim/optimizer.py)
//
// The port mirrors float64 operation order exactly (JS numbers are float64)
// and numba's comparison/min semantics (ternaries, never Math.min — NaN
// propagates differently). Parity is guarded by tests/js/optimizer.test.mjs
// against golden fixtures generated from the Python implementation.
// Change the Python and TypeScript sides together.

export interface CarParams {
  lat_limit: number;
  mass: number;
  acc_limit: number;
  dec_limit: number;
  c_drag: number;
  c_roll: number;
  corner_acc: number;
  trail_braking: number;
  P_engine_in_watt: number;
}

export interface TrackGeometry {
  width: Float64Array; // idx_len
  pos_min: Float64Array; // idx_len
  pos_max: Float64Array; // idx_len
  is_circular: boolean;
  idx_len: number;
  n_coords: number;
  left_coords: Float64Array; // flat n_coords * 3
  right_coords: Float64Array; // flat n_coords * 3
  slope: Float64Array; // idx_len
}

export interface AnnealState {
  position: Float64Array; // idx_len, normalized 0..1 across track width
  heatmap: Float64Array; // idx_len
  best_time: number;
  progress_rate: number;
}

export interface BatchResult {
  laptimes: Float64Array;
  positions: Float64Array[];
  perturbations: Float64Array[];
}

const G = 9.81;
export const MAX_DEVIATION = 0.1;
export const MAX_DEVIATION_LENGTH = 60;
export const F_ANNEAL = Math.pow(0.01, 1 / 10000); // from 1 to 0.01 in 10000 iterations without improvement
const HEATMAP_DECAY = 0.00015;

/**
 * Port of simulate_kernel + calc_speed (forward acceleration pass, reverse
 * braking pass), returning the laptime (sum of dt).
 */
export function simulateLaptime(lineCoordinates: Float64Array, slope: Float64Array, car: CarParams): number {
  const n = lineCoordinates.length / 3;
  const dX = new Float64Array(n * 3);
  const ddX = new Float64Array(n * 3);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 3; j++) {
      if (i === 0) {
        dX[i * 3 + j] = lineCoordinates[3 + j] - lineCoordinates[j];
      } else if (i === n - 1) {
        dX[i * 3 + j] = lineCoordinates[(n - 1) * 3 + j] - lineCoordinates[(n - 2) * 3 + j];
      } else {
        dX[i * 3 + j] = 0.5 * (lineCoordinates[(i + 1) * 3 + j] - lineCoordinates[(i - 1) * 3 + j]);
      }
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < 3; j++) {
      if (i === 0) {
        ddX[i * 3 + j] = dX[3 + j] - dX[j];
      } else if (i === n - 1) {
        ddX[i * 3 + j] = dX[(n - 1) * 3 + j] - dX[(n - 2) * 3 + j];
      } else {
        ddX[i * 3 + j] = 0.5 * (dX[(i + 1) * 3 + j] - dX[(i - 1) * 3 + j]);
      }
    }
  }

  const ds = new Float64Array(n);
  const kLat = new Float64Array(n);
  const gLon = new Float64Array(n);
  const gLat = new Float64Array(n);
  const vMax = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const i3 = i * 3;
    const m = Math.sqrt(dX[i3] * dX[i3] + dX[i3 + 1] * dX[i3 + 1] + dX[i3 + 2] * dX[i3 + 2]);
    ds[i] = m;
    const tx = dX[i3] / m;
    const ty = dX[i3 + 1] / m;
    const tz = dX[i3 + 2] / m;

    let bx = dX[i3 + 1] * ddX[i3 + 2] - dX[i3 + 2] * ddX[i3 + 1];
    let by = dX[i3 + 2] * ddX[i3] - dX[i3] * ddX[i3 + 2];
    let bz = dX[i3] * ddX[i3 + 1] - dX[i3 + 1] * ddX[i3];
    const bm = Math.sqrt(bx * bx + by * by + bz * bz);
    const k = bm / (m * m * m);
    bx /= bm;
    by /= bm;
    bz /= bm;

    const nx = by * tz - bz * ty;
    const ny = bz * tx - bx * tz;
    const nz = bx * ty - by * tx;
    const Nkx = nx * k;
    const Nky = ny * k;
    const Nkz = nz * k;

    // Rotate T 90deg CW in the xy-plane and align with the track slope
    let btx = ty;
    let bty = -tx;
    let btz = slope[i];
    const btm = Math.sqrt(btx * btx + bty * bty + btz * btz);
    btx /= btm;
    bty /= btm;
    btz /= btm;

    gLon[i] = G * tz;
    gLat[i] = G * btz;
    kLat[i] = Nkx * btx + Nky * bty + Nkz * btz;

    const sgn = kLat[i] > 0 ? 1.0 : kLat[i] < 0 ? -1.0 : 0.0;
    let kk = Math.abs(kLat[i]);
    if (kk < 1e-3) kk = 1e-3;
    let num = car.lat_limit - gLat[i] * sgn;
    if (num < 0) num = 0.0;
    vMax[i] = Math.sqrt(num / kk);
  }

  // forward pass for acceleration
  const vA = new Float64Array(n);
  calcSpeed(vA, ds, kLat, gLon, gLat, vMax, car.mass, car.acc_limit, car.lat_limit, car.c_drag, car.c_roll, car.corner_acc, car.P_engine_in_watt);

  // reverse pass for braking: reversed views, negated drag/roll/gravity, zero power
  const rev = (a: Float64Array) => {
    const r = new Float64Array(n);
    for (let i = 0; i < n; i++) r[i] = a[n - 1 - i];
    return r;
  };
  const gLonNeg = new Float64Array(n);
  for (let i = 0; i < n; i++) gLonNeg[i] = -gLon[n - 1 - i];
  const vBRev = new Float64Array(n);
  calcSpeed(
    vBRev,
    rev(ds),
    rev(kLat),
    gLonNeg,
    rev(gLat),
    rev(vMax),
    car.mass,
    car.dec_limit,
    car.lat_limit,
    -car.c_drag,
    -car.c_roll,
    car.trail_braking,
    0.0,
  );
  const vB = new Float64Array(n);
  for (let i = 0; i < n; i++) vB[i] = vBRev[n - 1 - i];

  const speed = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    speed[i] = vA[i] < vB[i] ? vA[i] : vB[i];
  }

  let laptime = 0;
  for (let i = 0; i < n; i++) {
    const im = i > 0 ? i - 1 : n - 1; // np.roll(speed, 1) semantics
    const dt = (2.0 * ds[i]) / (speed[i] + speed[im]);
    laptime += dt;
  }
  return laptime;
}

/** Port of calc_speed. speed[i] with negative i wraps like numpy (i + n). */
function calcSpeed(
  speed: Float64Array,
  ds: Float64Array,
  kLat: Float64Array,
  gLon: Float64Array,
  gLat: Float64Array,
  vMax: Float64Array,
  mass: number,
  limit: number,
  latLimit: number,
  cDrag: number,
  cRoll: number,
  factor: number,
  pEngineInWatt: number,
): void {
  const n = vMax.length;
  const nf = factor / 50;
  for (let i = -90; i < n - 1; i++) {
    const j = i + 1;
    const ii = i < 0 ? i + n : i;
    const jj = j < 0 ? j + n : j;

    // if corner speed was maximal, all grip is used for lateral acceleration
    if (!(speed[ii] < vMax[ii])) {
      speed[jj] = speed[ii] < vMax[jj] ? speed[ii] : vMax[jj];
      continue;
    }

    const v = speed[ii];
    // grip circle (no downforce accounted for)
    const accLat = v * v * kLat[ii] + gLat[ii];
    const maxAccGrip = limit * Math.pow(1 - Math.pow(Math.abs(accLat) / latLimit, nf), 1 / nf);

    // not the most readable code, but it mirrors the Python exactly
    const forceEngine = v !== 0 ? pEngineInWatt / v : 0;
    let accLon = forceEngine !== 0 ? (maxAccGrip < forceEngine / mass ? maxAccGrip : forceEngine / mass) : maxAccGrip;

    const aeroDrag = ((v * v) * cDrag) / 2 / mass;
    const rollingDrag = cRoll * G;
    accLon -= aeroDrag + rollingDrag + gLon[ii];

    const candidate = Math.sqrt(v * v + 2 * accLon * ds[ii]);
    speed[jj] = candidate < vMax[jj] ? candidate : vMax[jj];
  }
}

/** Port of _batch_kernel: builds, clips, interpolates and simulates candidates. */
export function buildBatch(
  geometry: TrackGeometry,
  currentPosition: Float64Array,
  locations: number[],
  lengths: number[],
  deviations: number[],
  car: CarParams,
): BatchResult {
  const m = locations.length;
  const idxLen = geometry.idx_len;
  const laptimes = new Float64Array(m);
  const positions: Float64Array[] = [];
  const perturbations: Float64Array[] = [];
  const coords = new Float64Array(geometry.n_coords * 3);

  for (let k = 0; k < m; k++) {
    let length = lengths[k];
    const location = locations[k];
    const deviation = deviations[k];
    const perturbation = new Float64Array(idxLen);

    const lineAdjust = new Float64Array(length);
    for (let a = 0; a < length; a++) {
      const t = length === 1 ? -Math.PI : -Math.PI + a * ((2 * Math.PI) / (length - 1));
      lineAdjust[a] = (1 + Math.cos(t)) / 2;
    }

    if (geometry.is_circular) {
      const shift = location - Math.floor(length / 2);
      for (let a = 0; a < length; a++) {
        perturbation[(((a + shift) % idxLen) + idxLen) % idxLen] = lineAdjust[a];
      }
    } else {
      if (location + length > idxLen) {
        length = idxLen - location;
      }
      for (let a = 0; a < length; a++) {
        perturbation[location + a] = lineAdjust[a];
      }
    }

    const newLine = new Float64Array(idxLen);
    for (let j = 0; j < idxLen; j++) {
      let v = currentPosition[j] + (perturbation[j] * deviation) / geometry.width[j];
      if (v < geometry.pos_min[j]) {
        v = geometry.pos_min[j];
      } else if (v > geometry.pos_max[j]) {
        v = geometry.pos_max[j];
      }
      newLine[j] = v;
    }

    for (let i = 0; i < idxLen; i++) {
      for (let j = 0; j < 3; j++) {
        coords[i * 3 + j] =
          geometry.left_coords[i * 3 + j] +
          (geometry.right_coords[i * 3 + j] - geometry.left_coords[i * 3 + j]) * newLine[i];
      }
    }
    if (geometry.is_circular) {
      for (let j = 0; j < 3; j++) {
        coords[(geometry.n_coords - 1) * 3 + j] = coords[j];
      }
    }

    laptimes[k] = simulateLaptime(coords, geometry.slope, car);
    positions.push(newLine);
    perturbations.push(perturbation);
  }
  return { laptimes, positions, perturbations };
}

/** Port of _apply_outcome: keep the best improving candidate, decay otherwise. */
export function applyOutcome(
  state: AnnealState,
  laptimes: Float64Array,
  positions: Float64Array[],
  perturbations: Float64Array[],
): boolean {
  let winner = -1;
  let min = Infinity;
  for (let k = 0; k < laptimes.length; k++) {
    const t = laptimes[k];
    if (!Number.isNaN(t) && t < min) {
      min = t;
      winner = k;
    }
  }

  if (winner >= 0) {
    const improvement = state.best_time - laptimes[winner];
    if (Number.isFinite(improvement) && improvement > 0) {
      state.position = positions[winner].slice();
      state.best_time = laptimes[winner];
      for (let i = 0; i < state.heatmap.length; i++) {
        state.heatmap[i] += perturbations[winner][i] * improvement * 1e3;
      }
      state.progress_rate += improvement;
      return true;
    }
  }

  // If the new line is not better, slowly decay the heatmap and progress rate
  for (let i = 0; i < state.heatmap.length; i++) {
    state.heatmap[i] = (state.heatmap[i] + HEATMAP_DECAY) / (1 + HEATMAP_DECAY);
  }
  state.progress_rate *= F_ANNEAL;
  return false;
}

/**
 * Port of anneal_step's sampling: heatmap-CDF weighted locations, random
 * lengths and deviations, then buildBatch + applyOutcome.
 */
export function annealStep(
  state: AnnealState,
  geometry: TrackGeometry,
  car: CarParams,
  batchSize: number,
  rng: () => number,
): BatchResult {
  const idxLen = geometry.idx_len;
  const cdf = new Float64Array(idxLen);
  let acc = 0;
  for (let i = 0; i < idxLen; i++) {
    acc += state.heatmap[i];
    cdf[i] = acc;
  }

  const locations: number[] = [];
  const lengths: number[] = [];
  const deviations: number[] = [];
  for (let k = 0; k < batchSize; k++) {
    const u = rng() * cdf[idxLen - 1];
    locations.push(Math.min(upperBound(cdf, u), idxLen - 1));
    lengths.push(3 + Math.floor(rng() * (MAX_DEVIATION_LENGTH - 3 + 1))); // integers(3, 60, endpoint=True)
    deviations.push((rng() * 2 - 1) * Math.max(lengths[k] / MAX_DEVIATION_LENGTH, MAX_DEVIATION));
  }

  const batch = buildBatch(geometry, state.position, locations, lengths, deviations, car);
  applyOutcome(state, batch.laptimes, batch.positions, batch.perturbations);
  return batch;
}

/** np.searchsorted(cdf, u, side="right"): count of elements <= u. */
function upperBound(cdf: Float64Array, u: number): number {
  let lo = 0;
  let hi = cdf.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cdf[mid] <= u) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}
