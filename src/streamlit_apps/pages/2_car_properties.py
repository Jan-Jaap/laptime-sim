"""This module creates a streamlit app"""

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

import laptime_sim
from laptime_sim.car import CornerAcceleration, DriverExperience
from main import CAR_LIST

G = 9.81  # m/s²


def plot_car_lon(race_car: laptime_sim.Car, v1):
    v = np.arange(300)
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot([race_car.get_acceleration(v=v0 / 3.6, acc_lat=0) / G for v0 in v], v, label="acc")
    ax.plot([-race_car.get_deceleration(v=v0 / 3.6, acc_lat=0) / G for v0 in v], v, label="dec")
    ax.axvline(0, color="grey", lw=1)
    ax.axhline(v1, color="grey", lw=1)
    ax.set_xlabel("Longitudinal acceleration (g)")
    ax.set_ylabel("Velocity (km/h)")
    ax.legend()
    return fig


def plot_car_lat(race_car: laptime_sim.Car, v1):
    x = np.linspace(-race_car.lat_limit, race_car.lat_limit, 100)

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(x / G, [race_car.get_acceleration(v=v1 / 3.6, acc_lat=lat) / G for lat in x], label="acc")
    ax.plot(x / G, [-race_car.get_deceleration(v=v1 / 3.6, acc_lat=lat) / G for lat in x], label="dec")
    ax.axvline(0, color="grey", lw=1)
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel("Lateral acceleration (g)")
    ax.set_ylabel("Longitudinal acceleration (g)")
    ax.legend()
    return fig


def main() -> None:
    st.set_page_config(page_title="HSR Webracing", layout="wide")
    st.header("Display car properties")

    race_car = st.radio(label="select file", options=CAR_LIST, format_func=lambda x: x.name)

    race_car.trail_braking = st.selectbox(
        label="Trailbraking driver experience",
        options=DriverExperience,
        index=list(DriverExperience).index(race_car.trail_braking),
        format_func=lambda x: DriverExperience(x).name,
    )

    race_car.corner_acc = st.selectbox(
        label="Select corner acceleration",
        options=CornerAcceleration,
        index=list(CornerAcceleration).index(race_car.corner_acc),
        format_func=lambda x: CornerAcceleration(x).name,
    )

    v1 = st.slider("Velocity in km/h", min_value=0, max_value=300)

    col1, col2 = st.columns(2)
    col1.pyplot(plot_car_lon(race_car, v1))
    col2.pyplot(plot_car_lat(race_car, v1))

    with st.expander("Car parameters"):
        st.write(race_car)


if __name__ == "__main__":
    main()
