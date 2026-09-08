# AGENTS.md

## Commands

- Setup: `uv sync` (Python 3.14 pinned in `.python-version`). The project itself is **not** installed (`[tool.uv] package = false`); imports are resolved via pytest `pythonpath = ["src"]` and Docker `PYTHONPATH=/app/src`.
- Tests: `uv run pytest` — **must run from repo root**, see below.
- Single test: `uv run pytest tests/test_track.py -k name`
- Streamlit app: `uv run streamlit run src/streamlit_apps/Welcome.py` (multipage; pages auto-discovered from `src/streamlit_apps/pages/`)
- Docker: `docker compose up --build -d` → http://localhost:8501

No typecheck or CI config exists. Lint/format with ruff (dev dep; config in `pyproject.toml`): `uv run ruff check --fix` then `uv run ruff format` on `src tests`; `src/temp/` is excluded.

## cwd requirement

`laptime_sim.registry` resolves `./resources/...` **relative to the current working directory**, and everything that loads resources (CLI, streamlit pages, tests) goes through it. Run all commands from the repo root, else loading fails with `FileNotFoundError`.

## Test suite

`registry.cars()`/`registry.tracks()` return `list[Car]`/`list[Track]` objects. Pytest collects only `tests/` (`testpaths` in pyproject); `src/temp/` scratch files are not tests.

## Architecture

- `src/main.py` — CLI entrypoint: simulated-annealing brute-force raceline optimization over every car × track. Saves one parquet per combo to `resources/simulated/` (gitignored) and **resumes from an existing file** if present, so deleting it restarts optimization from scratch. Resources and result files are handled via `laptime_sim.registry` inside `main()`.
- `src/laptime_sim/` — core library: `Track` (geopandas parquet; reprojects to UTM on init; borders tagged via a `geom_type` column with values `left`/`right`, written by streamlit page `3_create_track.py`), `Car` (dataclass, loaded from TOML in `resources/cars/`, power given in hp), `Raceline` (stores the line as a normalized 0–1 position across track width per point — not coordinates; use `coordinates_from_position` to convert), `registry` (single place for resource loading **and all result-file IO**: `results_path`/`save_result`/`load_result`/`read_results` — streamlit pages and tests import from here, not from `main`; page 3's track creation is exempt/unfinished).
- `simulate.py` — physics core; hot loops are numba `@njit(cache=True)`. First run compiles (~seconds), afterwards the numba cache in `__pycache__` makes runs fast. `tests/test_benchmark.py` reports ms/iteration (`uv run pytest tests/test_benchmark.py -s`); `tests/test_car_parity.py` guards `Car.get_acceleration`/`get_deceleration` against drifting from the physics inline in `calc_speed`.
- `optimizer.py` — batch candidate evaluation: `run_batch()` builds, clips, interpolates and simulates chunks of random candidate lines inside a `nogil` numba kernel on a thread pool (`N_THREADS`); `raceline.try_random_lines()` samples candidates and keeps the best.
- `Track.initial_line()`/`centerline()` are reserved for the upcoming track-creation flow (page 3) — not dead code, even though they currently have no callers.
- `src/temp/` — untracked scratch code (gitignored); ignore it.

## Conventions

- New car = new TOML file in `resources/cars/` (dataclass `Car` fields). New track = parquet created via streamlit page 3 in `resources/tracks/`.
- Result filenames are built by `registry.results_path(car, track)` from `Car.file_name` (name with spaces → underscores) + track name; changing `Car.name` orphans old result files.
