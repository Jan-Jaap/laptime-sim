"""This module creates a streamlit app"""

import streamlit as st
from matplotlib import pyplot as plt

import laptime_sim
from laptime_sim import Car, Raceline, Track
from laptime_sim.optimizer import optimize
from laptime_sim.registry import cars, load_result, results_path, save_result, tracks


@st.cache_resource
def get_cars() -> list[Car]:
    return cars()


@st.cache_resource
def get_tracks() -> list[Track]:
    return tracks()


def optimize_raceline(track: Track, car: Car, raceline: Raceline):
    if "optimization_running" not in st.session_state:
        st.session_state["optimization_running"] = False

    with st.status("Raceline optimization", state="error", expanded=True) as status:
        placeholder_saved = st.empty()
        placeholder_laptime = st.empty()
        if st.button("Start/Stop - Optimize raceline"):
            if not st.session_state.optimization_running:  # if not running start te optimization
                st.session_state.optimization_running = True
                status.update(state="running")
                placeholder_saved.write("optimization is started")

                placeholder_laptime.write(f"Laptime = {raceline.best_time_str()}")
                timer = laptime_sim.Timer()
                for update in optimize(
                    raceline,
                    car,
                    save_every=30.0,
                    on_save=lambda: save_result(raceline, car),
                ):
                    placeholder_laptime.write(
                        f"Laptime = {update['best_time_str']}. "
                        f"candidates:{update['candidates']}, "
                        f"candidate_rate:{update['candidates'] / timer.elapsed_time.total_seconds():.0f}"
                    )
                    if update["finished"]:
                        status.update(state="complete")
                        placeholder_saved.write(f"Raceline optimization converged: {update['best_time_str']}")
                        break

            if st.session_state.optimization_running:  # if running stop te optimization
                st.session_state.optimization_running = False
                placeholder_saved.write("optimization is stopped")

        save_result(raceline, car)


def main() -> None:
    with st.sidebar:
        track = st.radio("select track", options=get_tracks(), format_func=lambda x: x.name)
        race_car = st.radio("select car", options=get_cars(), format_func=lambda x: x.name)

    st.header(f"Racetrack - {track.name}")

    file_path = results_path(race_car, track)
    raceline, found = load_result(track, race_car)

    if found:
        st.warning(f"Filename {file_path.name} exists and will be overwritten")
    else:
        st.warning("No results found for this car. A new raceline will be created.")

    raceline.simulate(race_car)
    st.info(f"{race_car.name}, Best time = {raceline.best_time_str()}")

    optimize_raceline(track, race_car, raceline)

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
