"use strict";
(() => {
  // src/streamlit_apps/components/raceline_optimizer/src/optimizer.ts
  var G = 9.81;
  var MAX_DEVIATION = 0.1;
  var MAX_DEVIATION_LENGTH = 60;
  var F_ANNEAL = Math.pow(0.01, 1 / 1e4);
  var HEATMAP_DECAY = 15e-5;
  function simulateLaptime(lineCoordinates, slope, car) {
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
      const sgn = kLat[i] > 0 ? 1 : kLat[i] < 0 ? -1 : 0;
      let kk = Math.abs(kLat[i]);
      if (kk < 1e-3) kk = 1e-3;
      let num = car.lat_limit - gLat[i] * sgn;
      if (num < 0) num = 0;
      vMax[i] = Math.sqrt(num / kk);
    }
    const vA = new Float64Array(n);
    calcSpeed(vA, ds, kLat, gLon, gLat, vMax, car.mass, car.acc_limit, car.lat_limit, car.c_drag, car.c_roll, car.corner_acc, car.P_engine_in_watt);
    const rev = (a) => {
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
      0
    );
    const vB = new Float64Array(n);
    for (let i = 0; i < n; i++) vB[i] = vBRev[n - 1 - i];
    const speed = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      speed[i] = vA[i] < vB[i] ? vA[i] : vB[i];
    }
    let laptime = 0;
    for (let i = 0; i < n; i++) {
      const im = i > 0 ? i - 1 : n - 1;
      const dt = 2 * ds[i] / (speed[i] + speed[im]);
      laptime += dt;
    }
    return laptime;
  }
  function calcSpeed(speed, ds, kLat, gLon, gLat, vMax, mass, limit, latLimit, cDrag, cRoll, factor, pEngineInWatt) {
    const n = vMax.length;
    const nf = factor / 50;
    for (let i = -90; i < n - 1; i++) {
      const j = i + 1;
      const ii = i < 0 ? i + n : i;
      const jj = j < 0 ? j + n : j;
      if (!(speed[ii] < vMax[ii])) {
        speed[jj] = speed[ii] < vMax[jj] ? speed[ii] : vMax[jj];
        continue;
      }
      const v = speed[ii];
      const accLat = v * v * kLat[ii] + gLat[ii];
      const maxAccGrip = limit * Math.pow(1 - Math.pow(Math.abs(accLat) / latLimit, nf), 1 / nf);
      const forceEngine = v !== 0 ? pEngineInWatt / v : 0;
      let accLon = forceEngine !== 0 ? maxAccGrip < forceEngine / mass ? maxAccGrip : forceEngine / mass : maxAccGrip;
      const aeroDrag = v * v * cDrag / 2 / mass;
      const rollingDrag = cRoll * G;
      accLon -= aeroDrag + rollingDrag + gLon[ii];
      const candidate = Math.sqrt(v * v + 2 * accLon * ds[ii]);
      speed[jj] = candidate < vMax[jj] ? candidate : vMax[jj];
    }
  }
  function buildBatch(geometry, currentPosition, locations, lengths, deviations, car) {
    const m = locations.length;
    const idxLen = geometry.idx_len;
    const laptimes = new Float64Array(m);
    const positions = [];
    const perturbations = [];
    const coords = new Float64Array(geometry.n_coords * 3);
    for (let k = 0; k < m; k++) {
      let length = lengths[k];
      const location = locations[k];
      const deviation = deviations[k];
      const perturbation = new Float64Array(idxLen);
      const lineAdjust = new Float64Array(length);
      for (let a = 0; a < length; a++) {
        const t = length === 1 ? -Math.PI : -Math.PI + a * (2 * Math.PI / (length - 1));
        lineAdjust[a] = (1 + Math.cos(t)) / 2;
      }
      if (geometry.is_circular) {
        const shift = location - Math.floor(length / 2);
        for (let a = 0; a < length; a++) {
          perturbation[((a + shift) % idxLen + idxLen) % idxLen] = lineAdjust[a];
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
        let v = currentPosition[j] + perturbation[j] * deviation / geometry.width[j];
        if (v < geometry.pos_min[j]) {
          v = geometry.pos_min[j];
        } else if (v > geometry.pos_max[j]) {
          v = geometry.pos_max[j];
        }
        newLine[j] = v;
      }
      for (let i = 0; i < idxLen; i++) {
        for (let j = 0; j < 3; j++) {
          coords[i * 3 + j] = geometry.left_coords[i * 3 + j] + (geometry.right_coords[i * 3 + j] - geometry.left_coords[i * 3 + j]) * newLine[i];
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
  function applyOutcome(state, laptimes, positions, perturbations) {
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
    for (let i = 0; i < state.heatmap.length; i++) {
      state.heatmap[i] = (state.heatmap[i] + HEATMAP_DECAY) / (1 + HEATMAP_DECAY);
    }
    state.progress_rate *= F_ANNEAL;
    return false;
  }
  function annealStep(state, geometry, car, batchSize, rng) {
    const idxLen = geometry.idx_len;
    const cdf = new Float64Array(idxLen);
    let acc = 0;
    for (let i = 0; i < idxLen; i++) {
      acc += state.heatmap[i];
      cdf[i] = acc;
    }
    const locations = [];
    const lengths = [];
    const deviations = [];
    for (let k = 0; k < batchSize; k++) {
      const u = rng() * cdf[idxLen - 1];
      locations.push(Math.min(upperBound(cdf, u), idxLen - 1));
      lengths.push(3 + Math.floor(rng() * (MAX_DEVIATION_LENGTH - 3 + 1)));
      deviations.push((rng() * 2 - 1) * Math.max(lengths[k] / MAX_DEVIATION_LENGTH, MAX_DEVIATION));
    }
    const batch = buildBatch(geometry, state.position, locations, lengths, deviations, car);
    applyOutcome(state, batch.laptimes, batch.positions, batch.perturbations);
    return batch;
  }
  function upperBound(cdf, u) {
    let lo = 0;
    let hi = cdf.length;
    while (lo < hi) {
      const mid = lo + hi >> 1;
      if (cdf[mid] <= u) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  }

  // src/streamlit_apps/components/raceline_optimizer/src/rng.ts
  function mulberry32(seed) {
    let a = seed >>> 0;
    return () => {
      a = a + 1831565813 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  // src/streamlit_apps/components/raceline_optimizer/src/worker.ts
  var TOLERANCE = 0.1;
  function yieldToEventLoop() {
    return new Promise((resolve) => {
      const channel = new MessageChannel();
      channel.port1.onmessage = () => {
        channel.port1.close();
        resolve();
      };
      channel.port2.postMessage(null);
    });
  }
  async function runAnnealing(state, geometry, car, opts) {
    const started = performance.now();
    let candidates = 0;
    let lastProgress = 0;
    for (let step = 0; !opts.shouldStop(); step++) {
      annealStep(state, geometry, car, opts.batchSize, opts.rng);
      candidates += opts.batchSize;
      const now = performance.now();
      if (now - lastProgress >= opts.progressIntervalMs) {
        lastProgress = now;
        const elapsed = (now - started) / 1e3;
        opts.onProgress({
          type: "progress",
          best_time: state.best_time,
          candidates,
          rate: candidates / elapsed,
          progress_rate: state.progress_rate,
          finished: false
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
      finished: true
    });
  }
  function toGeometry(g) {
    return {
      width: Float64Array.from(g.width),
      pos_min: Float64Array.from(g.pos_min),
      pos_max: Float64Array.from(g.pos_max),
      is_circular: g.is_circular,
      idx_len: g.idx_len,
      n_coords: g.n_coords,
      left_coords: Float64Array.from(g.left_coords.flat()),
      right_coords: Float64Array.from(g.right_coords.flat()),
      slope: Float64Array.from(g.slope)
    };
  }
  function coordsFromPosition(geometry, position) {
    const coords = new Float64Array(geometry.n_coords * 3);
    for (let i = 0; i < geometry.idx_len; i++) {
      for (let j = 0; j < 3; j++) {
        coords[i * 3 + j] = geometry.left_coords[i * 3 + j] + (geometry.right_coords[i * 3 + j] - geometry.left_coords[i * 3 + j]) * position[i];
      }
    }
    if (geometry.is_circular) {
      for (let j = 0; j < 3; j++) {
        coords[(geometry.n_coords - 1) * 3 + j] = coords[j];
      }
    }
    return coords;
  }
  var stopped = false;
  async function startRun(msg) {
    const geometry = toGeometry(msg.geometry);
    const car = msg.car;
    const state = {
      position: Float64Array.from(msg.position),
      heatmap: new Float64Array(geometry.idx_len).fill(1),
      best_time: msg.best_time,
      progress_rate: 1
    };
    if (!Number.isFinite(state.best_time)) {
      state.best_time = simulateLaptime(coordsFromPosition(geometry, state.position), geometry.slope, car);
    }
    await runAnnealing(state, geometry, car, {
      batchSize: msg.batch_size,
      rng: mulberry32(msg.seed),
      progressIntervalMs: msg.progress_interval_ms,
      shouldStop: () => stopped,
      onProgress: (m) => postMessage(m),
      onResult: (m) => postMessage(m)
    });
  }
  if (typeof self !== "undefined") {
    self.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === "stop") {
        stopped = true;
      } else if (msg.type === "start") {
        stopped = false;
        void startRun(msg);
      }
    };
  }
})();
