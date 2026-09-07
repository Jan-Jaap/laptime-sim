# AGENTS.md

## Commands

- Setup: `uv sync` (Python 3.12 pinned in `.python-version`). The project itself is **not** installed (`[tool.uv] package = false`); imports are resolved via pytest `pythonpath = ["src"]` and Docker `PYTHONPATH=/app/src`.
- Tests: `uv run pytest` — **must run from repo root**, see below.
- Single test: `uv run pytest tests/test_track.py -k name`
- Streamlit app: `uv run streamlit run src/streamlit_apps/Welcome.py` (multipage; pages auto-discovered from `src/streamlit_apps/pages/`)
- Docker: `docker compose up --build -d` → http://localhost:8501

No lint, typecheck, or CI config exists.

## Import-time cwd requirement

`src/main.py` loads `./resources/cars/` and `./resources/tracks/` **at import time** using relative paths, and the tests import `main`. Anything importing `main` — including pytest — must run from the repo root, else it fails with `FileNotFoundError`.

## Test suite

`CAR_LIST`/`TRACK_LIST` in `main.py` are already `list[Car]`/`list[Track]` objects — do **not** pass them back into `car_list()`/`track_list()` (those take directory paths). Pytest collects only `tests/` (`testpaths` in pyproject); `src/temp/` scratch files are not tests.

## Architecture

- `src/main.py` — CLI entrypoint: simulated-annealing brute-force raceline optimization over every car × track. Saves one parquet per combo to `resources/simulated/` (gitignored) and **resumes from an existing file** if present, so deleting it restarts optimization from scratch.
- `src/laptime_sim/` — core library: `Track` (geopandas parquet; reprojects to UTM on init; borders tagged via a `geom_type` column with values `left`/`right`, written by streamlit page `3_create_track.py`), `Car` (pydantic, loaded from TOML in `resources/cars/`, power given in hp), `Raceline` (stores the line as a normalized 0–1 position across track width per point — not coordinates; use `coordinates_from_position` to convert).
- `simulate.py` — physics core; hot loops are numba `@njit`, so the first call in a session pays JIT compilation (slow first sim/test run — not a hang).
- `src/temp/` — untracked scratch code (gitignored); ignore it.

## Conventions

- New car = new TOML file in `resources/cars/` (pydantic `Car` fields). New track = parquet created via streamlit page 3 in `resources/tracks/`.
- Result filenames are built from `Car.file_name` (name with spaces → underscores) + track name; changing `Car.name` orphans old result files.
