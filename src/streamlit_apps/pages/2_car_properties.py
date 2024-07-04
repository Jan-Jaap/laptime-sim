"""This module creates a streamlit app"""

from pathlib import Path

import numpy as np
import plotly.express as px
import streamlit as st

import laptime_sim
from laptime_sim.car import CornerAcceleration, Trailbraking
from laptime_sim.main import get_all_cars

G = 9.81  # m/s²
PATH_CARS = Path("./cars/")

def plot_car_lon(race_car: laptime_sim.Car, v1):
    v = np.linspace(0, 300, 100)
    fig = px.line(
        dict(
            v=v,
            acc=[race_car.get_acceleration(v=v0 / 3.6, acc_lat=0) / G for v0 in v],
            dec=[-race_car.get_deceleration(v=v0 / 3.6, acc_lat=0) / G for v0 in v],
        ),
        x=["acc", "dec"],
        y="v",
    )
    fig.add_vline(x=0)
    fig.add_hline(y=v1)
    return fig


def plot_car_lat(race_car: laptime_sim.Car, v1):

    x = np.linspace(-race_car.acc_grip_max, race_car.acc_grip_max, 100)

    fig = px.line(
        dict(
            x=x / G,
            acc=[race_car.get_acceleration(v=v1 / 3.6, acc_lat=lat) / G for lat in x],
            dec=[-race_car.get_deceleration(v=v1 / 3.6, acc_lat=lat) / G for lat in x],
        ),
        x="x",
        y=["acc", "dec"],
    )
    fig.add_vline(x=0)
    fig.add_vline(x=race_car.acc_grip_max / G)
    fig.add_vline(x=-race_car.acc_grip_max / G)
    fig.add_hline(y=0)
    fig.add_hline(y=0)
    return fig


def main() -> None:
    st.set_page_config(page_title="HSR Webracing", layout="wide")
    st.header("Display car properties")

    race_car = st.radio(label="select file", options=get_all_cars(PATH_CARS), format_func=lambda x: x.name)

    race_car.trail_braking = st.selectbox(
        label="Trailbraking driver experience",
        options=Trailbraking,
        index=list(Trailbraking).index(race_car.trail_braking),
        format_func=lambda x: Trailbraking(x).name,
    )

    race_car.corner_acc = st.selectbox(
        label="Select corner acceleration",
        options=CornerAcceleration,
        index=list(CornerAcceleration).index(race_car.corner_acc),
        format_func=lambda x: CornerAcceleration(x).name,
    )

    v1 = st.slider("Velocity in km/h", min_value=0, max_value=300)

    col1, col2 = st.columns(2)
    col1.plotly_chart(
        plot_car_lon(race_car, v1),
        use_container_width=True,
    )
    col2.plotly_chart(
        plot_car_lat(race_car, v1),
        use_container_width=True,
    )

    with st.expander("Car parameters"):
        st.write(race_car)


if __name__ == "__main__":
    main()
