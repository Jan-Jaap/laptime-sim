"""This module creates a streamlit app"""

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

from laptime_sim import Car, Raceline, Track, format_laptime
from laptime_sim._browser_component import get_raceline_optimizer
from laptime_sim.optimizer import accept_result
from laptime_sim.raceline import BORDER_CLEARANCE_M
from laptime_sim.registry import cars, load_result, results_path, tracks

BATCH_SIZE = 200
PROGRESS_INTERVAL_MS = 500


@st.cache_resource
def get_cars() -> list[Car]:
    return cars()


@st.cache_resource
def get_tracks() -> list[Track]:
    return tracks()


def _track_payload(track: Track, raceline: Raceline) -> dict:
    clearance = BORDER_CLEARANCE_M / raceline.width
    return {
        "width": raceline.width.tolist(),
        "pos_min": clearance.tolist(),
        "pos_max": (1 - clearance).tolist(),
        "is_circular": bool(track.is_circular),
        "idx_len": len(raceline.width),
        "n_coords": track.len,
        "left_coords": track.left_coords.tolist(),
        "right_coords": track.right_coords.tolist(),
        "slope": track.slope.tolist(),
    }


def _car_payload(car: Car) -> dict:
    return {
        "lat_limit": car.lat_limit,
        "mass": car.mass,
        "acc_limit": car.acc_limit,
        "dec_limit": car.dec_limit,
        "c_drag": car.c_drag,
        "c_roll": car.c_roll,
        "corner_acc": int(car.corner_acc),
        "trail_braking": int(car.trail_braking),
        "P_engine_in_watt": car.P_engine_in_watt,
    }


def _get_raceline(track: Track, race_car: Car) -> tuple[Raceline, bool]:
    """Loads the stored raceline once per track/car selection (session-cached):
    component value updates rerun this page, and a parquet read per rerun would
    make every one of them expensive."""
    key = (track.name, race_car.name)
    if st.session_state.get("raceline_key") != key:
        raceline, found = load_result(track, race_car)
        raceline.simulate(race_car)
        st.session_state.raceline = raceline
        st.session_state.raceline_found = found
        st.session_state.raceline_key = key
        # the component keeps its last value across reruns — without a reset the
        # previous track's result would be accepted against the new track
        st.session_state.optimization_running = False
        st.session_state.run_id = st.session_state.get("run_id", 0) + 1
        st.session_state.processed_result = None
    return st.session_state.raceline, st.session_state.raceline_found


def _optimize_in_browser(track: Track, race_car: Car, raceline: Raceline) -> None:
    """Runs the annealing loop on the visitor's CPU via a custom component.

    Live progress stays inside the component iframe (zero streamlit reruns
    while optimizing); only the final result is posted back, and
    `accept_result` re-simulates it server-side before saving."""
    if "optimization_running" not in st.session_state:
        st.session_state.optimization_running = False
        st.session_state.run_id = 0
        st.session_state.processed_result = None

    if st.button("Start/Stop optimization (runs on your CPU)"):
        st.session_state.optimization_running = not st.session_state.optimization_running
        if st.session_state.optimization_running:
            # run_id identifies a run (start->stop cycle): the worker echoes it
            # with its result, and bumping it on stop would make the stale-guard
            # reject that very result
            st.session_state.run_id += 1
        st.session_state.processed_result = None

    raceline_optimizer = get_raceline_optimizer()
    value = raceline_optimizer(
        track_payload=_track_payload(track, raceline),
        car_payload=_car_payload(race_car),
        position=raceline.position.tolist(),
        best_time=float(raceline.best_time),
        batch_size=BATCH_SIZE,
        seed=st.session_state.run_id,
        progress_interval_ms=PROGRESS_INTERVAL_MS,
        running=st.session_state.optimization_running,
        run_id=st.session_state.run_id,
        key="raceline_optimizer",
    )

    if st.session_state.optimization_running:
        st.caption("Optimizing on your CPU — live progress is shown inside the component box above.")
    else:
        st.caption("Press start — the optimization then runs in your browser, not on the server.")

    if value is None:
        return

    if value["type"] == "error":
        st.error(f"Optimizer failed: {value['message']}")
        return

    if value["type"] == "result":
        if value["run_id"] != st.session_state.run_id:
            return  # stale echo from before a selection change or restart
        result_key = f"{value['run_id']}:{value['candidates']}"
        if st.session_state.processed_result == result_key:
            return
        st.session_state.processed_result = result_key
        try:
            accepted, improved = accept_result(track, race_car, np.asarray(value["position"], dtype=np.float64))
        except ValueError as exc:  # invalid candidate must not crash the page
            st.error(f"Rejected optimizer result: {exc}")
            return
        st.session_state.optimization_running = False  # run is over (stopped or converged)
        if improved:
            st.success(f"New best laptime: {format_laptime(accepted.best_time)} (saved)")
            st.session_state.raceline = accepted  # improved line becomes the next baseline
            st.session_state.raceline_found = True
        else:
            st.info(f"No improvement over the stored line ({format_laptime(accepted.best_time)}).")


def main() -> None:
    with st.sidebar:
        track = st.radio("select track", options=get_tracks(), format_func=lambda x: x.name)
        race_car = st.radio("select car", options=get_cars(), format_func=lambda x: x.name)

    st.header(f"Racetrack - {track.name}")

    file_path = results_path(race_car, track)
    raceline, found = _get_raceline(track, race_car)

    if found:
        st.warning(f"Filename {file_path.name} exists and will be overwritten")
    else:
        st.warning("No results found for this car. A new raceline will be created.")

    st.info(f"{race_car.name}, Best time = {raceline.best_time_str()}")

    _optimize_in_browser(track, race_car, raceline)

    with st.expander("Race Car"):
        st.write(race_car)
    with st.expander("Selected Raceline"):
        sim_results = raceline.simulate(race_car)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sim_results.distance, sim_results.speed_kph)
        ax.set_ylabel("Speed in km/h")
        ax.set_xlabel("Track distance in m")
        st.pyplot(fig)
        st.write(raceline)
    with st.expander("SimResults"):
        st.write(sim_results)


if __name__ == "__main__":
    main()
