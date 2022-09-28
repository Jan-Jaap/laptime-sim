import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import altair as alt

from threading import Thread
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

from track_sim.sim import Track


def page_config():
    st.set_page_config(
        page_title='HSR Webracing',
        layout='wide',

        )

def plot_track_plotly(track: Track, line=None):

    MODE = 'lines'
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=track.left_x, y=track.left_y,
        mode=MODE, name='left side' ))
    fig.add_trace(go.Scatter(x=track.right_x, y=track.right_y,
        mode=MODE, name='right side' ))
    if line is not None:
        fig.add_trace(go.Scatter(x=line[:,0], y=line[:,1],
            mode=MODE, name='line', line=dict(width=2, dash='dash')))
    fig.update_xaxes(showticklabels=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, zeroline=False, scaleanchor = "x", scaleratio = 1)

    st.plotly_chart(fig, use_container_width=True)

def plot_track_matplotlib(track: Track, line=None):
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.plot(track.left_x, track.left_y)
    plt.plot(track.right_x, track.right_y)
    plt.plot(line[:,0], line[:,1])
    
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    ax.set_aspect('equal', 'box')

    fig.tight_layout()
    st.pyplot(fig)

def plot_track_altair(track: Track, line=None):
    source = track.get_track_borders()
    fig = alt.Chart(source).mark_circle(size=10).encode(
        x=alt.X('left_x', scale = alt.Scale(domain=[min(track.left_x), max(track.left_x)])),
        y=alt.Y('left_y', scale = alt.Scale(domain=[min(track.left_y), max(track.left_y)])),
        )
    st.altair_chart(fig, use_container_width=True)


def plot_track(track: Track, line=None):
    return plot_track_plotly(track, line)



def write(str):
    st.write(str)

def input(str):
    return st.checkbox(str)


def run_thread(thread: Thread):

    add_script_run_ctx(thread)
    if st.checkbox('start worker'):
        thread.start()
    elif thread is not None:
        print(f'{get_script_run_ctx()} should stop')
        thread.should_stop = True


def upload_track():
    return st.file_uploader('Upload results file', 'csv')