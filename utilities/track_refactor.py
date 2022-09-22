import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from track_sim.sim import Track

i = 800

NAME_TRACK = '20191211_bilsterberg_sparse'
CIRCULAR_TRACK = False

file_out = f'./tracks/{NAME_TRACK}_{i}.csv'

df_track = pd.read_csv(f'./tracks/{NAME_TRACK}.csv')

outer3d = ['outer_x','outer_y','outer_z']
inner3d = ['inner_x','inner_y', 'inner_z']

race_track = Track(
    name = NAME_TRACK,
    border_left=df_track[outer3d].values,
    border_right=df_track[inner3d].values,
    )

curvature = race_track.get_curvature(0.5)
s = np.abs(curvature) + 0.025
s = s.cumsum()

if CIRCULAR_TRACK:
    # ensure first and last track coordinate are overlapping
    if sum(df_track[outer3d].iloc[0].values - df_track[outer3d].iloc[-1].values) != 0:
        df_track.iloc[-1] = df_track.iloc[0]
    bc_type = 'periodic'   
else:
    bc_type = 'not-a-knot'

#reparameterize track coordinates based on track curvature
boundary = CubicSpline(s, df_track[outer3d+inner3d].values, bc_type=bc_type)
boundary = boundary(np.linspace(boundary.x.min(), boundary.x.max(), i))

#save results
pd.DataFrame(boundary, columns=outer3d+inner3d).to_csv(file_out, index=None)
           
