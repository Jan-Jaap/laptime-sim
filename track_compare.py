# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:53:14 2019

@author: JV
"""


import pandas

#import matplotlib.pyplot as plt


c_drag = 0.5 * 0.34 * 1.22 * 1.78 / 1050


df = pandas.read_csv('./BMW Z3M Viperwizard_Zandvoort_simulated.csv')
#df2 = pandas.read_csv('./logged_data/session_zandvoort_circuit_20190930_1445_v1.csv', skiprows = 10) # BMW Z3M
df2 = pandas.read_csv('./logged_data/NO_BMWZ3M/session_zandvoort_circuit_20190930_2045_v2.csv', skiprows = 10) #205 GTi RFS

df2 = df2[df2['Lap #']==2]
df2['Distance (m)'] -= df2['Distance (m)'].values[0]


#import plotly_express as px
import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default = 'browser'

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Lateral acceleration (m/s2)'], y=df['Longitudinal acceleration (m/s2)']))
fig.add_trace(go.Scatter(x=df2['Lateral acceleration (m/s2)'], y=df2['Longitudinal acceleration (m/s2)']))


fig.show()

# =============================================================================
# plt.figure()
# plt.plot(df['Distance (m)'], df['Speed (m/s)']*3.6)
# plt.plot(df2['Distance (m)'], df2['Speed (m/s)']*3.6)
# 
# plt.figure()
# plt.plot(df['Lateral acceleration (m/s2)'], df['Longitudinal acceleration (m/s2)'])
# plt.plot(df2['Lateral acceleration (m/s2)'], df2['Longitudinal acceleration (m/s2)'])
# =============================================================================


