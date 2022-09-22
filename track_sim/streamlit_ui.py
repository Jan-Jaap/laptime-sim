import streamlit as st
import plotly.graph_objects as go

from threading import Thread
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx

MODE = 'markers+lines'

def page_config():
    st.set_page_config(
        page_title='HSR Webracing',
        layout='wide',

        )
    

def plot_track(track, line=None):

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


def upload_results():
    return st.file_uploader('Upload results file', 'csv')