import os
import numpy as np
import plotly.express as px
import streamlit as st

import car
import race_lap

G = 9.81  # m/s²


def st_car_ggv_display(path_cars):

    files_in_dir = [f for f in sorted(os.listdir(path_cars)) if f.endswith('toml')]
    filename_car = os.path.join(path_cars, st.radio(label="select file", options=files_in_dir))

    race_car = car.Car.from_toml(filename_car)

    col1, col2 = st.columns(2)

    f = st.selectbox('Trailbraking driver experience', car.Trailbraking._member_names_, index=3)
    race_car.trail_braking = st.slider('Trail braking', min_value=30, max_value=100, value=car.Trailbraking[f])

    f = st.selectbox('Select corner acceleration', car.CornerAcceleration._member_names_, index=3)
    race_car.corner_acc = st.slider(
        label='Corner acceleration',
        min_value=30,
        max_value=100,
        value=car.CornerAcceleration[f]
        )

    v = np.linspace(0, 300, 100)
    v1 = st.slider('Velocity in km/h', min_value=v.min(), max_value=v.max())

    fig = px.line(dict(
        v=v,
        acc=[race_lap.get_max_acceleration(race_car, v=v0/3.6, acc_lat=0)/G for v0 in v],
        dec=[-race_lap.get_max_deceleration(race_car, v=v0/3.6, acc_lat=0)/G for v0 in v],
        ), x=['acc', 'dec'], y='v')
    fig.add_vline(x=0)
    fig.add_hline(y=v1)
    col1.plotly_chart(fig, use_container_width=True, )

    x = np.linspace(-race_car.acc_grip_max, race_car.acc_grip_max, 100)

    fig = px.line(dict(
        x=x/G,
        acc=[race_lap.get_max_acceleration(race_car, v=v1/3.6, acc_lat=lat)/G for lat in x],
        dec=[-race_lap.get_max_deceleration(race_car, v=v1/3.6, acc_lat=lat)/G for lat in x],
        ), x='x', y=['acc', 'dec'])
    fig.add_vline(x=0)
    fig.add_vline(x=race_car.acc_grip_max/G)
    fig.add_vline(x=-race_car.acc_grip_max/G)
    fig.add_hline(y=0)
    fig.add_hline(y=0)
    col2.plotly_chart(fig, use_container_width=True, )

    with st.expander('Car parameters'):
        st.write(race_car)
