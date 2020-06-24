# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas
import numpy as np 
from psopy import _minimize_pso, init_feasible
from scipy.signal import savgol_filter
from scipy.stats import norm
import matplotlib.pyplot as plt


#@profile
def LapSim1(df, race_line):
    
    Raceline_x = df.Left_x.values + (df.Right_x.values - df.Left_x.values) * race_line
    Raceline_y = df.Left_y.values + (df.Right_y.values - df.Left_y.values) * race_line
#    df['Raceline_z'] = df.Left_z + (df.Right_z - df.Left_z) * race_line
    if True: #smoothing
        Raceline_x = savgol_filter(Raceline_x, 61, 3) # window size 51, polynomial order 3
        Raceline_y = savgol_filter(Raceline_y, 61, 3) # window size 51, polynomial order 3
    
    dx = np.diff(Raceline_x, prepend=Raceline_x[-1])
    dy = np.diff(Raceline_y, prepend=Raceline_y[-1])
    
    Path_S = (dx**2 + dy**2)**0.5
    Path_angle = np.arctan2(dx,dy)
    
    dA = np.diff(Path_angle, prepend=Path_angle[-1])
    
    Curvature = np.abs(np.tan(dA) / Path_S).clip(1e-3)
    v_max = (max_lat_acc / Curvature)**0.5

        
    v_sim_acc = np.zeros(np.size(v_max))
    v_sim_dec = np.zeros(np.size(v_max))
    v_sim_acc[0] = v_max[0]
    v_sim_dec[0] = v_max[-1]
    
    
    for i in range(1, len(df.index)):
        v_sim_acc[i] =   min( (v_sim_acc[i-1] * v_sim_acc[i-1] + 2*max_lon_acc * Path_S[i])**0.5,  v_max[i])  
        v_sim_dec[i] =   min( (v_sim_dec[i-1] * v_sim_dec[i-1] + 2*min_lon_acc * Path_S[-i-1])**0.5,  v_max[-i-1])  
    
    v_sim_dec = v_sim_dec[::-1] #flip te matrix
    v_sim = np.minimum(v_sim_acc, v_sim_dec)


    return [Path_S, dA, Curvature, v_max, v_sim_acc, v_sim_dec, v_sim] 

def new_Raceline(Raceline, mean, stdev, mag):
    x = np.arange(len(Raceline))+1
    x05 = len(Raceline)/2
    y = norm.pdf(np.roll(x,int(mean - x05)),  x05 , stdev) * stdev * mag
    Raceline += y
    return Raceline.clip(0.05,0.95)

def calc_LapTime(df):
    return df.Path_S.sum() / df.v_sim.mean()

def fun(x1):
    s=[]
    for x in x1:
        raceline = new_Raceline(df.Path_position.copy().values,x[0], x[1], x[2])
        [Path_S, dA, Curvature, v_max, v_sim_acc, v_sim_dec, v_sim] = LapSim1(df, raceline)

        cost = Path_S.sum() / v_sim.mean()  #optimise laptime only
#        cost = Path_S.sum() / v_sim.mean() + (dA**2).sum()  #last part is input penalty to avoid swerving
#        cost =  (dA**2).sum()  #minimum input change
        s.append(cost)
    return s

if __name__ == '__main__':

    max_lat_acc = 1.15 * 9.81
    max_lon_acc = 0.15 * 9.81
    min_lon_acc = 10
    
    # read starting positions from file
    df = pandas.read_csv('WP_POS_simulated_v2.csv', index_col='Row Labels')
    
    s = calc_LapTime(LapTimeSim(df, df.Path_position.values))
    print('Simulated laptime = {:02.0f}:{:05.02f}'.format(s%3600//60, s%60))
 
    
try:
    while True:

        x0 = np.zeros([10,3])
        x0[:,0] = np.random.uniform(0, len(df.index),len(x0))
        x0[:,1] = np.random.uniform(0, 50,len(x0))
        x0[:,2] = np.random.randn(len(x0))/100
        
        res = _minimize_pso(fun,x0, stable_iter=10)
        print('Optimized cost = {} - {}'.format(res.fun, res.x))
        
#        df.Path_position = new_Raceline(df.Path_position.values, res.x[0], res.x[1], res.x[2])
        df.Path_position = new_Raceline(df.Path_position.values, res.x[0], res.x[1], res.x[2])
        s = calc_LapTime(LapTimeSim(df, df.Path_position.values))
        print('Optimized laptime = {:02.0f}:{:05.02f}'.format(s%3600//60, s%60))


except KeyboardInterrupt:
    
    s = calc_LapTime(LapTimeSim(df, df.Path_position))
    print('Simulated laptime = {:02.0f}:{:05.02f}'.format(s%3600//60, s%60))
#%% save df            
    df.to_csv('WP_POS_simulated_v2.csv')
#%% plot results    
    plt.figure()
    plt.axis('equal')

    plt.plot(df.Left_x, df.Left_y)    
    plt.plot(df.Right_x, df.Right_y)
    plt.plot(df.Raceline_x, df.Raceline_y)

    
    
    plt.figure()
    plt.plot( df['v_max_m/s'])
    plt.plot(df.v_sim_acc)
    plt.plot(df.v_sim_dec)
    plt.plot(df.v_sim)
    


       


    