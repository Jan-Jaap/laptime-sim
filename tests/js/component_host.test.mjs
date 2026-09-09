// Smoke test for main.js component-host logic without a browser:
// mocks window.parent.postMessage + Worker + a minimal DOM, drives render
// events, and asserts the component handshake (including the mandatory
// isStreamlitMessage flag), the worker start payload, and that live progress
// stays INSIDE the iframe (no setComponentValue reruns) — only the final
// result is sent to Python.
//
// Run: node tests/js/component_host.test.mjs   (part of `npm test`)

import assert from "node:assert";

const posted = [];
let messageListener = null;
const dom = { byId: new Map(), created: [] };

globalThis.window = {
  parent: { postMessage: (m) => posted.push(m) },
  addEventListener: (_type, fn) => {
    messageListener = fn;
  },
};
globalThis.document = {
  body: {
    appendChild: (el) => dom.byId.set(el.id, el),
    style: {},
    scrollHeight: 60,
  },
  getElementById: (id) => dom.byId.get(id) ?? null,
  createElement: () => {
    const el = { style: {}, textContent: "" };
    dom.created.push(el);
    return el;
  },
};
globalThis.Worker = class {
  constructor(src) {
    posted.push({ type: "worker-created", src });
    globalThis.__worker = this;
  }
  postMessage(m) {
    posted.push({ kind: "worker-msg", ...m });
  }
  terminate() {}
};

await import("../../src/streamlit_apps/components/raceline_optimizer/main.js");

assert(messageListener, "main.js registered a message listener");
// the streamlit host drops any message whose data lacks isStreamlitMessage
assert(posted.length > 0 && posted.every((m) => m.isStreamlitMessage === true), "every posted message carries isStreamlitMessage");
assert(
  posted.some((m) => m.type === "streamlit:componentReady" && m.apiVersion === 1),
  "componentReady posted",
);

const args = {
  track_payload: { width: [1, 2], idx_len: 2, is_circular: true },
  car_payload: { mass: 1000 },
  position: [0.5, 0.5],
  best_time: 100,
  batch_size: 8,
  seed: 3,
  progress_interval_ms: 5,
  running: true,
  run_id: 1,
};
messageListener({ data: { type: "streamlit:render", args } });

const created = posted.find((m) => m.type === "worker-created");
assert(created && created.src === "worker.js", "worker created from relative worker.js");
const start = posted.find((m) => m.kind === "worker-msg");
assert(start && start.geometry && start.car && start.batch_size === 8, "worker received start payload");

// stop flow: rerun with running=false
messageListener({ data: { type: "streamlit:render", args: { ...args, running: false } } });
// worker got a stop message (the last worker-msg after the start)
const stopMsg = posted.filter((m) => m.kind === "worker-msg").at(-1);
assert(stopMsg && stopMsg.type === "stop", "stop forwarded to worker");

// live progress must stay inside the iframe: DOM updates, no setComponentValue
messageListener({ data: { type: "streamlit:render", args: { ...args, running: true, run_id: 2 } } });
const worker = globalThis.__worker;
const before = posted.filter((m) => m.type === "streamlit:setComponentValue").length;
worker.onmessage({
  data: { type: "progress", best_time: 118.5, candidates: 1200, rate: 300, progress_rate: 0.9 },
});
const progressEl = dom.byId.get("progress");
assert(progressEl, "progress element created in the iframe");
assert(progressEl.textContent.includes("optimizing"), "progress text rendered");
assert(progressEl.textContent.includes("1:58.50"), "formatted laptime rendered");
assert(
  posted.filter((m) => m.type === "streamlit:setComponentValue").length === before,
  "progress must not trigger a streamlit rerun",
);

// the final result is the only value posted to Python
worker.onmessage({
  data: { type: "result", position: [0.5, 0.5], best_time: 118.5, candidates: 1200, finished: true },
});
const values = posted.filter((m) => m.type === "streamlit:setComponentValue");
assert(values.length === before + 1, "exactly one setComponentValue for the result");
assert(values.at(-1).value.type === "result", "the posted value is the result");
assert(values.at(-1).value.run_id === 2, "the result echoes the run's run_id");

// the rerun following the posted result: iframe reports "result sent", no new value
messageListener({ data: { type: "streamlit:render", args: { ...args, running: false, run_id: 2 } } });
assert(progressEl.textContent.includes("result sent"), "iframe confirms the result was sent");
assert(
  posted.filter((m) => m.type === "streamlit:setComponentValue").length === before + 1,
  "no extra value after the result was processed",
);

// pressing start after a finished run must restart the worker (no restart on stale renders)
messageListener({ data: { type: "streamlit:render", args: { ...args, running: false, run_id: 2 } } });
messageListener({ data: { type: "streamlit:render", args: { ...args, running: true, run_id: 3 } } });
const restarted = globalThis.__worker;
assert(restarted !== worker, "a new run starts a new worker");

console.log("main.js smoke passed");
