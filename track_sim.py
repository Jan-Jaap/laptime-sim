# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas
import numpy as np 
from psopy import _minimize_pso, init_feasible
from scipy.stats import norm
import matplotlib.pyplot as plt

def LapTimeSim(df, race_line):
     
    df['Raceline_x'] = df.Left_x + (df.Right_x - df.Left_x) * race_line
    df['Raceline_y'] = df.Left_y + (df.Right_y - df.Left_y) * race_line
    df['Raceline_z'] = df.Left_z + (df.Right_z - df.Left_z) * race_line
    
    dx = df.Raceline_x.diff()
    dx[1] = df.Raceline_x[1] - df.Raceline_x.iloc[-1] 
    dy = df.Raceline_y.diff()
    dy[1] = df.Raceline_y[1] - df.Raceline_y.iloc[-1] 
    
    df['Path_S'] = (dx**2 + dy**2)**0.5
    df['Path_angle'] = np.arctan2(dx,dy)%(2*np.pi)
    
    dA = df.Path_angle.diff()
    dA[1] = df.Path_angle[1] - df.Path_angle.iloc[-1]
    dA.loc[:] = (dA + np.pi) % (2*np.pi) - np.pi
    
    df['Curvature'] = np.tan(dA) / df.Path_S
    df['v_max_m/s'] = np.abs(max_lat_acc / df.Curvature)**0.5
    
    df['v_max_m/s'].loc[:] = df['v_max_m/s'].clip(upper=100)
    
    
    v_sim_acc = [df['v_max_m/s'].iloc[0]]
    v_sim_dec = [df['v_max_m/s'].iloc[-1]]
    
    for i in range(1, len(df.index)):
        v_sim_acc.append(   min(    (v_sim_acc[i-1]**2+2*max_lon_acc*df.Path_S.iloc[i])**0.5,    df['v_max_m/s'].iloc[i])  )
        v_sim_dec.append(   min(    (v_sim_dec[i-1]**2+2*min_lon_acc*df.Path_S.iloc[-i-1])**0.5,    df['v_max_m/s'].iloc[-i-1])  )
    df['v_sim_acc'] = v_sim_acc[:]
    df['v_sim_dec'] = v_sim_dec[::-1]
    
    df['v_sim'] = df[['v_sim_acc', 'v_sim_dec']].min(axis=1)[:]

    return df

#@profile
def LapTimeSim1(df, race_line):
     
    Raceline_x = df.Left_x.values + (df.Right_x.values - df.Left_x.values) * race_line.values
    Raceline_y = df.Left_y.values + (df.Right_y.values - df.Left_y.values) * race_line.values
#    df['Raceline_z'] = df.Left_z + (df.Right_z - df.Left_z) * race_line
    
    dx = np.diff(Raceline_x, prepend=Raceline_x[-1])
    dy = np.diff(Raceline_y, prepend=Raceline_y[-1])
    
    Path_S = (dx**2 + dy**2)**0.5
    Path_angle = np.arctan2(dx,dy)%(2*np.pi)
    
    dA = np.diff(Path_angle, prepend=Path_angle[-1])
#    dA[1] = Path_angle[1] - Path_angle.iloc[-1]
    dA = (dA + np.pi) % (2*np.pi) - np.pi
    
    Curvature = np.max(np.tan(dA) / Path_S, 1e-8)
    v_max = np.abs(max_lat_acc / Curvature)**0.5
    v_max = v_max.clip(0,100)
    
    
    v_sim_acc = np.zeros(np.size(v_max))
    v_sim_dec = np.zeros(np.size(v_max))
    v_sim_acc[0] = v_max[0]
    v_sim_dec[0] = v_max[0]
    
    
    for i in range(1, len(df.index)):
        v_sim_acc[i] = (   df.min(    (v_sim_acc[i-1]**2+2*max_lon_acc*Path_S[i])**0.5,    v_max[i])  )
        v_sim_dec[i] = (   df.min(    (v_sim_dec[i-1]**2+2*min_lon_acc*Path_S[-i-1])**0.5,    v_max[-i-1])  )
    
    v_sim = np.minimum(v_sim_acc, v_sim_dec)
    LapTime = Path_S.sum() / v_sim.mean()

    return LapTime 


def adjust_Raceline(Raceline0, mean, stdev, mag):
    Raceline1 = Raceline0.copy()
    x = Raceline1.index
    x05 = x.max()/2
    y = norm.pdf(np.roll(x,int(mean - x05)),  int(x05) , stdev)*stdev*mag
  
#    plt.plot(x,y)
       
    Raceline1 += y
    return Raceline1


def calc_LapTime(df):
    return df.Path_S.sum() / df.v_sim.mean()


def fun(df, x0):
    s=[]
    raceline0 = df.Path_position.copy()
    for x in x0:
        raceline = adjust_Raceline(raceline0,x[0], x[1], x[2])
        s.append(LapTimeSim1(df, raceline))
    return s

if __name__ == '__main__':

    max_lat_acc = 1.15 * 9.81
    max_lon_acc = 0.15 * 9.81
    min_lon_acc = 10
    
    df = pandas.read_csv('WP_POS.csv', index_col='Row Labels')
    #print(df)

    Raceline0 = df.Path_position
    df = LapTimeSim(df, Raceline0)
    s = calc_LapTime(df)
    print('Simulated laptime = {:02.0f}:{:02.2f}'.format(s%3600//60, s%60))
#        
#    
#    Raceline1 = adjust_Raceline(df.Path_position, 1000, 10, -1)
#    s = calc_LapTime(LapTimeSim(df, Raceline1))
#    print('Simulated laptime = {:02.0f}:{:02.2f}'.format(s%3600//60, s%60))

    x0 = np.random.rand(10,3)
    x0[:,0] *= df.index.max()
    x0[:,1] *= 2
    x0[:,2] -= 0.5; x0[:,2] *= 1
    
    s1 = fun(df,x0)
    
#    print('Simulated laptime = {:02.0f}:{:02.2f}'.format(s%3600//60, s%60))
   

    
    df.to_csv('WP_POS_simulated.csv')


