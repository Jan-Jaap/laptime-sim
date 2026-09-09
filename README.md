# Laptime Sim Python

A simple and fast laptime simulator for determining the laptime when racing optimal speed.

It uses an approximate implementation of the algoritm described in this paper.
http://jameshakewill.com/Lap_Time_Simulation.pdf

I'm no math wizard. Optimization of laptimes is done by bruteforce trying different lines and using the fastest line found. By trying thousands of different options, we can come close (enough) to the optimum raceline. Some would call this AI...


## Installation
Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to create python virtual environment and install dependencies:

There is a streamlit app to visualize the results.
```bash
uv run streamlit run src/streamlit_apps/Welcome.py
```

Also a docker compose file to run the streamlit app in docker is provided.
```
docker compose up --build -d
```
This will run the app on [http://localhost:8501](http://localhost:8501)

## Usage
Run the CLI optimizer to optimize all cars on all tracks:

```bash
uv run laptime-sim
```

The streamlit optimization page (page 4) runs the raceline optimization in the visitor's browser (TypeScript kernels in a Web Worker) — the server only serves tracks/cars and persists results.

## Development

Python dependencies and linting are managed with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync                      # install dependencies (installs this project editable too)
uv run pytest                # run the test suite
uv run ruff check --fix src tests
uv run ruff format src tests
```

The browser optimizer (streamlit page 4) is written in TypeScript and bundled with [esbuild](https://esbuild.github.io/); it requires [Node.js](https://nodejs.org/) (>= 22):

```bash
npm ci                       # once, after cloning
npm run build                # bundle src/worker.ts -> committed worker.js
npm run typecheck            # tsc --noEmit
npm test                     # parity tests against the Python kernels + component tests
```

The built `worker.js` is committed to the repo, so running the app does not require Node. If you change the Python physics (`simulate.py`/`optimizer.py`), regenerate the golden parity fixtures and rebuild:

```bash
uv run python tests/js/generate_golden.py
npm run build && npm test
```

## Examples of usage

![streamlit_trackview](/resources/images/streamlit_trackview.png)

![streamlit_car_properties](/resources/images/streamlit_car_properties.png)

