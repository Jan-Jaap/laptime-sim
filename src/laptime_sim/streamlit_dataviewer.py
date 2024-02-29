"""This module creates a streamlit app"""

import streamlit as st

from st_racetrack_display import st_racetrack_display
from st_car_ggv_display import st_car_ggv_display

PATH_TRACKS = "./tracks/"
PATH_LINES = "./simulated/"
PATH_CARS = "./cars/"


def app():
    st.set_page_config(page_title="HSR Webracing", layout="wide")

    tab1, tab2 = st.tabs(['Track definitions', 'Car definitions'])
    with tab1:
        st_racetrack_display(path_tracks=PATH_TRACKS, path_racelines=PATH_LINES)
    with tab2:
        st_car_ggv_display(path_cars=PATH_CARS)


if __name__ == "__main__":
    app()
