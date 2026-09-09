// Vanilla-JS Streamlit custom component hosting the optimizer Web Worker.
//
// Protocol (see streamlit-component-lib):
//   child -> parent: {type: "streamlit:componentReady", apiVersion: 1, isStreamlitMessage: true}
//   parent -> child: {type: "streamlit:render", args: {...}, isStreamlitMessage: true}
//   child -> parent: {type: "streamlit:setComponentValue", value, dataType: "json", isStreamlitMessage: true}
// The host drops any message whose data lacks `isStreamlitMessage: true` —
// omitting it silently disables the whole component.
//
// Python args:
//   track_payload, car_payload, position, best_time, batch_size, seed,
//   progress_interval_ms, running (bool), run_id (increments per start/stop)
//
// Live progress is rendered INSIDE this iframe (the worker posts ~2 Hz, the
// DOM is updated directly) — progress must never reach setComponentValue,
// because every component value triggers a full streamlit script rerun. Only
// the final {type: "result"} is posted to Python.
//
// A render only (re)starts the worker when running=true and the payload changed
// (run_id), so start/stop toggles work without resetting a running optimization.

(() => {
  let worker = null;
  let running = false;
  let lastRunKey = null;
  let activeRunId = null;
  let textColor = null;
  let resultSent = false;

  const post = (message) => window.parent.postMessage({ isStreamlitMessage: true, ...message }, "*");

  const ensureProgressEl = () => {
    let el = document.getElementById("progress");
    if (!el) {
      el = document.createElement("div");
      el.id = "progress";
      el.style.cssText = "font:500 13px/1.7 'Source Sans Pro',sans-serif; padding:8px 2px; white-space:nowrap; overflow:hidden;";
      document.body.appendChild(el);
    }
    if (textColor) el.style.color = textColor;
    return el;
  };

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const rest = seconds - minutes * 60;
    return `${minutes}:${rest.toFixed(2).padStart(5, "0")}`;
  };

  const setProgressText = (text) => {
    ensureProgressEl().textContent = text;
    post({ type: "streamlit:setFrameHeight", height: Math.max(40, document.body.scrollHeight) });
  };

  const startWorker = (args) => {
    if (worker) worker.terminate();
    worker = new Worker("worker.js");
    running = true;
    activeRunId = args.run_id;
    setProgressText("optimizing — starting kernels…");
    worker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === "progress") {
        setProgressText(
          `optimizing — best ${formatTime(msg.best_time)} · ` +
            `${msg.candidates.toLocaleString()} candidates · ${Math.round(msg.rate)}/s`,
        );
      } else if (msg.type === "result") {
        running = false;
        resultSent = true;
        setProgressText(
          `finished — best ${formatTime(msg.best_time)} · ${msg.candidates.toLocaleString()} candidates — sending result…`,
        );
        // the ONLY value posted to Python: one rerun carrying the final result
        setValue({ ...msg, run_id: activeRunId });
      }
    };
    worker.onerror = (event) => {
      running = false;
      setProgressText(`error: ${event.message || "worker error"}`);
      setValue({ type: "error", message: event.message || "worker error", run_id: activeRunId });
    };
    worker.postMessage({
      type: "start",
      geometry: args.track_payload,
      car: args.car_payload,
      position: args.position,
      best_time: args.best_time,
      batch_size: args.batch_size,
      seed: args.seed,
      progress_interval_ms: args.progress_interval_ms,
    });
  };

  const stopWorker = () => {
    running = false;
    setProgressText("stopping…");
    if (worker) worker.postMessage({ type: "stop" });
  };

  const setValue = (value) => post({ type: "streamlit:setComponentValue", value, dataType: "json" });

  window.addEventListener("message", (event) => {
    const msg = event.data;
    if (msg.type !== "streamlit:render") return;
    const args = msg.args || {};
    if (typeof args.running === "undefined" || !args.track_payload) return;

    if (msg.theme && msg.theme.textColor) textColor = msg.theme.textColor;

    const runKey = `${args.run_id}:${JSON.stringify(args.track_payload)}:${JSON.stringify(args.car_payload)}`;
    if (args.running && !running) {
      // fresh start (run_id changed) or restart after a finished run
      lastRunKey = runKey;
      resultSent = false;
      startWorker(args);
    } else if (args.running && running && runKey !== lastRunKey) {
      // payload changed mid-run (new track/car): restart
      lastRunKey = runKey;
      startWorker(args);
    } else if (!args.running && running) {
      // the worker posts its final result after stopping
      stopWorker();
    } else if (!args.running && resultSent) {
      // the rerun following the posted result: the server's verdict appears below the component
      resultSent = false;
      setProgressText("result sent — see status below");
    }
  });

  post({ type: "streamlit:componentReady", apiVersion: 1 });
})();
