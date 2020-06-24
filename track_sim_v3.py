# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas
import numpy as np 
from psopy import _minimize_pso, init_feasible
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import norm
import matplotlib.pyplot as plt


#@profile
def LapTimeSim1(df, raceline=None, full_results=False):

    if not hasattr(raceline, '__len__'):
        raceline = df['Path_position']
    
    Raceline_x = df.Left_x.values + (df.Right_x.values - df.Left_x.values) * raceline
    Raceline_y = df.Left_y.values + (df.Right_y.values - df.Left_y.values) * raceline
    if True: #smoothing
        Raceline_x = savgol_filter(Raceline_x, 101, 5, mode='wrap') # window size 51, polynomial order 2
        Raceline_y = savgol_filter(Raceline_y, 101, 5, mode='wrap') # window size 51, polynomial order 2
        
    
    dx = np.diff(Raceline_x, prepend=Raceline_x[-1])
    dy = np.diff(Raceline_y, prepend=Raceline_y[-1])
    
    Path_S = (dx**2 + dy**2)**0.5
    Path_angle = np.arctan2(dx,dy)
    dA = np.diff(Path_angle, append=Path_angle[0])  #yaw speed
    
    Curvature = np.abs(np.tan(dA) / Path_S).clip(1e-3)
    v_max = (max_lat_acc / Curvature)**0.5


    #build extended index to account for start and stop velocity (hotlap)
    i = len(v_max)
    
    v_sim_acc = np.zeros(i)
    v_sim_dec = np.zeros(i)
    
    for i in range(-800,i):  #- index to simulate running start....
        
        if v_sim_acc[i-1] < v_max[i-1]:
            acc_lat = v_sim_acc[i-1]**2 * Curvature[i-1]
            acc_lon = max_lon_acc * (max_lat_acc**2 - acc_lat**2)**0.5 / max_lat_acc
            dec_lon = max_lon_dec * (max_lat_acc**2 - acc_lat**2)**0.5 / max_lat_acc
        else:
            acc_lat = max_lat_acc
            acc_lon = 0
            dec_lon = 0

        v_sim_acc[i] =   min( (v_sim_acc[i-1] * v_sim_acc[i-1] + 2*acc_lon * Path_S[i])**0.5,  v_max[i])
        v_sim_dec[i] =   min( (v_sim_dec[i-1] * v_sim_dec[i-1] + 2*dec_lon * Path_S[-i-1])**0.5,  v_max[-i-1])  
  
    v_sim_dec = v_sim_dec[::-1] #flip te matrix
    v_sim = np.minimum(v_sim_acc, v_sim_dec)

    if full_results:
        df['Path_position'] = raceline
        df['Raceline_x'] = Raceline_x
        df['Raceline_y'] = Raceline_y
        df['Raceline_z'] = df.Left_z + (df.Right_z - df.Left_z) * raceline
        df['Path_S'] = Path_S
        df['dA'] = dA
        df['Curvature'] = Curvature
        df['v_max_m/s'] = v_max
        df['v_sim_acc'] = v_sim_acc
        df['v_sim_dec'] = v_sim_dec
        df['v_sim'] = v_sim
        return df
    else:
        return Path_S.sum() / v_sim.mean()
    
def new_Raceline(Raceline, x):
    [mean, stdev, mag] = x
    x = np.arange(len(Raceline))+1
    x05 = len(Raceline)/2
    y = norm.pdf(np.roll(x,int(mean - x05)),  x05 , stdev) * stdev * mag
    Raceline = (Raceline + y).clip(0.1,0.9)
    return Raceline


def fun(x1):
    #copy() is used to prevent net raceline to be saved in the dataframe
    s = [LapTimeSim1(df, new_Raceline(df.Path_position.values, x)) for x in x1]
    return s



if __name__ == '__main__':

    max_lat_acc = 1.05 * 9.81
    max_lon_acc = 0.15 * 9.81
    max_lon_dec = 10.0
    
    # read starting positions from file
    df = pandas.read_csv('WP_POS_simulated_v1.csv', index_col='Row Labels')
    
#    s = LapTimeSim1(df, df.Path_position.values)
    s = LapTimeSim1(df)
    print('Simulated laptime = {:02.0f}:{:05.02f}'.format(s%3600//60, s%60))
     
try:
    while True:

        i_acc = find_peaks(-df.v_sim, prominence=1)[0]+1
        x0 = np.zeros([len(i_acc),3])
        x0[:,0] = np.random.uniform(0, len(df.index),len(x0))
        x0[:,1] = np.random.uniform(0, 500,len(x0))
        x0[:,2] = np.random.randn(len(x0))/40
        
        res = _minimize_pso(fun,x0, stable_iter=10)
        print('Optimized cost = {} - {}'.format(res.fun, res.x))

        df.Path_position = new_Raceline(df.Path_position.values, res.x)  #passed not a copy, but path directly.
        LapTimeSim1(df, full_results=True)  #saves results in df
        s = LapTimeSim1(df) 
        print('Optimized laptime = {:02.0f}:{:05.02f}'.format(s%3600//60, s%60))


except KeyboardInterrupt:
    s = LapTimeSim1(df) #save df  
    print('Simulated laptime = {:02.0f}:{:05.02f}'.format(s%3600//60, s%60))



#%% save df            
    LapTimeSim1(df, full_results=True)
    df.to_csv('WP_POS_simulated_v1.csv')
    
    
    
#%% plot results    


#    i=find_peaks(df.v_sim, width=25)[0]
    i_acc = find_peaks(-df.v_sim, prominence=1)[0]+1
    i_dec = find_peaks(df.v_sim, prominence=1)[0]+1
    
    plt.figure()
    plt.axis('equal')
    plt.plot(df.Left_x, df.Left_y)    
    plt.plot(df.Right_x, df.Right_y)
    plt.plot(df.Raceline_x, df.Raceline_y)
    plt.plot(df.Raceline_x[i_acc], df.Raceline_y[i_acc], 'bo')
    plt.plot(df.Raceline_x[i_dec], df.Raceline_y[i_dec], 'ro')
    
    
    plt.figure()
    plt.plot(df['v_max_m/s'])
    plt.plot(df.v_sim_acc)
    plt.plot(df.v_sim_dec)
    plt.plot(df.v_sim)
    plt.plot(df.Path_distance[i_acc], df.v_sim[i_acc], 'bo')
    plt.plot(df.Path_distance[i_dec], df.v_sim[i_dec], 'ro')
