import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def display_track():
    match st.radio('select filetype', options=['tracks', 'simulation results']):
        case 'tracks':
            dir = './tracks/'
        case  'simulation results':
            dir = './simulated/'

    if filename_track := st.radio(label='select track', options=[s for s in sorted(os.listdir(dir)) if s.endswith('.csv')]):

        df_track = pd.read_csv(dir+filename_track)

        if 'Timestamp' in df_track.columns:
            laptime = df_track.Timestamp.iloc[-1]
            st.write(f'Simulated laptime = {laptime%3600//60:02.0f}:{laptime%60:06.03f}')
  
        border_left         = df_track.filter(regex="outer_").values
        border_right        = df_track.filter(regex="inner_").values
        race_line           = df_track.filter(regex="line_").values
        
        MODE = 'lines'
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=border_left[:,0] , y=border_left[:,1] ,mode=MODE, name='border_left' ))
        fig.add_trace(go.Scatter(x=border_right[:,0], y=border_right[:,1],mode=MODE, name='border_right' ))
        
        if race_line.size != 0:
            fig.add_trace(go.Scatter(x=race_line[:,0], y=race_line[:,1],
                mode=MODE, name='line', line=dict(width=2, dash='dash')))
        
        fig.update_xaxes(showticklabels=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, zeroline=False, scaleanchor = "x", scaleratio = 1)

        st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    st.set_page_config(
    page_title='HSR Webracing',
    layout='wide')

    tab1, tab2 = st.tabs(['Display track','Optimize raceline'])

    with tab1:
        st.header('Race track display')
        display_track()
