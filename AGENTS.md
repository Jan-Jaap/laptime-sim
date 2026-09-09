# AGENTS.md

## Commands

- Setup: `uv sync` (Python 3.14 pinned in `.python-version`). Installs dependencies **and the project itself** (editable), so `import laptime_sim` resolves in any shell — no `PYTHONPATH` setup needed locally.
- CLI optimizer: `uv run laptime-sim` — long-running by design: optimizes every car × track until convergence, autosaves every 10 s. Interrupt with Ctrl+C (results are saved on interrupt).
- Tests: `uv run pytest` (run from repo root so pytest discovers `testpaths`). ~10 s once the numba cache is warm.
- Single test: `uv run pytest tests/test_track.py -k name`
- Benchmark: `uv run pytest tests/test_benchmark.py -s` (ms/iteration). The first-ever run pays numba JIT compilation; the cache in `__pycache__` makes later runs fast.
- JS/TS (browser optimizer): `npm ci` once; `npm run build` (esbuild bundle → committed `worker.js`), `npm run typecheck`, `npm test` (parity tests vs Python). Regenerate golden fixtures with `uv run python tests/js/generate_golden.py` after changing the Python kernels.
- Streamlit app: `uv run streamlit run src/streamlit_apps/Welcome.py` (multipage; pages auto-discovered from `src/streamlit_apps/pages/`).
- Docker: `docker compose up --build -d` → http://localhost:8501 (the builder runs `uv sync --no-install-project` because `src/` isn't copied yet; the runtime image sets `PYTHONPATH=/app/src` — keep that flag if you touch the Dockerfile).

CI (`.github/workflows/ci.yml`) runs a python job (ruff check + ruff format --check + pytest) and a typescript job (npm typecheck + parity tests) on push/PR. Lint/format with ruff (dev dep; config in `pyproject.toml`): `uv run ruff check --fix` then `uv run ruff format` on `src tests`; `src/temp/` is excluded.

## Resource paths

`laptime_sim.registry` anchors `resources/...` to the package location (`_ROOT = Path(__file__).resolve().parents[2]`), so resource loading works from any cwd — locally (editable install) and in Docker (`/app`). Every load/save of cars, tracks, and results goes through it; `cars()`/`tracks()` accept a `path` override for tests, and the writers resolve their directories at call time. Result files land in `resources/simulated/` (gitignored).

## Test suite

- `registry.cars()`/`registry.tracks()` return `list[Car]`/`list[Track]` objects. Pytest collects only `tests/` (`testpaths` in pyproject); `src/temp/` scratch files are not tests. `tests/conftest.py` and some test modules still call `cars()`/`tracks()` at import time.
- `tests/test_streamlit_apps.py` imports every streamlit page module: pages must stay importable without a running streamlit server (page bodies load resources only inside `main()`/cached loaders).
- `tests/test_car_parity.py` guards `Car.get_acceleration`/`get_deceleration` (used by the UI) against drifting from the physics inline in `simulate.calc_speed` (used by the optimizer) — change both together.

## Architecture

- `src/laptime_sim/cli.py` — CLI entrypoint (`uv run laptime-sim`): simulated-annealing brute-force raceline optimization over every car × track via `optimizer.optimize()`. Saves one parquet per combo to `resources/simulated/` (gitignored); an existing file's line is **loaded and optimization resumes from it**, so deleting it restarts optimization from scratch.
- `src/laptime_sim/` — core library:
  - `Track` (geopandas parquet; reprojects to UTM on init; needs a `track_name` column and borders tagged via a `geom_type` column with values `left`/`right`).
  - `Car` (dataclass, loaded from TOML in `resources/cars/`, power given in hp).
  - `Raceline` (pure state: stores the line as a normalized 0–1 position across track width per point — not coordinates; use `coordinates_from_position` to convert — plus heatmap/progress bookkeeping).
  - `registry` (single place for resource loading **and all result/track-file IO**: `results_path`/`save_result`/`load_result`/`read_results`/`save_track` — streamlit pages and tests import from here).
- `simulate.py` — physics core; hot loops are numba `@njit(cache=True)`.
- `optimizer.py` — candidate evaluation **and strategy**: `run_batch()` builds, clips, interpolates and simulates chunks of random candidate lines inside a `nogil` numba kernel on a thread pool (`N_THREADS` = cpu count); `anneal_step()` samples candidates, accepts improvements and anneals the heatmap; `optimize()` is the shared convergence loop (generator yielding status dicts + autosave callbacks) consumed by the CLI; `accept_result()` is the client-result seam (validates, re-simulates server-side, saves only improvements — client laptime claims are never trusted). `Raceline.try_random_lines()` delegates to `anneal_step`.
- **Browser optimizer (page 4)**: optimization runs on the *visitor's* CPU via a Streamlit custom component (`src/streamlit_apps/components/raceline_optimizer/` — vanilla JS host + Web Worker) using a TypeScript port of the kernels (`src/optimizer.ts` mirrors `simulate_kernel`/`_batch_kernel`/`_apply_outcome` — change Python and TS together). Parity is guarded by golden fixtures in `tests/js/fixtures/` replayed by `tests/js/optimizer.test.mjs`; the component host logic by `tests/js/component_host.test.mjs` (both via `npm test`). The built `worker.js` bundle is committed so Docker needs no Node. `declare_component` must be called from `laptime_sim._browser_component`, never from a page: streamlit resolves the caller via `inspect.getmodule`, and exec'd pages are not in `sys.modules` (`RuntimeError: module is None`). Component assets are served at `/component/{registered_name}/{file}` — a 200 that returns the app shell HTML means the URL missed the component route. The host drops any postMessage whose data lacks `isStreamlitMessage: true`.
- `Track.initial_line()`/`centerline()` are reserved for the upcoming track-creation flow (page 3) — not dead code, even though they currently have no callers.
- Streamlit pages load car/track lists via `@st.cache_resource` loaders, never at module level; page 2 builds a `dataclasses.replace()` copy per selection instead of mutating the shared `Car` objects.
- `src/temp/` — untracked scratch code (gitignored); ignore it.

## Track creation (streamlit page 3) — unfinished

- Saves via `registry.save_track()` directly into `resources/tracks/`.
- Input is an uploaded **zipped shapefile** (polygon); the saved parquet gets the `track_name` + `geom_type` (`left`/`right`) columns `Track` requires.

## Conventions

- New car = new TOML file in `resources/cars/` using the `Car` dataclass fields as keys. Values are validated in `Car.__post_init__`, and `trail_braking` must be one of `DriverExperience` (30/50/70/100), `corner_acc` one of `CornerAcceleration` (50/60/80/90) — other 0–100 values raise `ValueError`.
- Result filenames are built by `registry.results_path(car, track)` from `Car.file_name` (name with spaces → underscores) + track name; changing `Car.name` orphans old result files (`read_results` filters them out).
