# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas
import numpy as np 
from numpy.linalg import norm
#from psopy import _minimize_pso, init_feasible
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import norm
from scipy.interpolate import CubicSpline
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt




class Raceline:
    def __init__(self,x,y):
        
        if x[-1] != x[0] or y[-1] != y[0]:
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            
        self.x = x
        dx = np.diff(x, prepend=x[0])
        self.y = y
        dy = np.diff(y, prepend=y[0])
        
        self.s = np.sum(np.c_[dx**2,dy**2], axis=1)**0.5
        s = cumtrapz(self.s, initial=0)
        self.cspline = CubicSpline(s, np.c_[x,y] , bc_type='periodic')
        

        return
        
    def simulate(self):
        s = self.cspline.x
        [dx, dy]    = self.cspline(s,1).T
        [ddx, ddy]  =  self.cspline(s,2).T
        k = (dx*ddy - dy*ddx)/(dx**2+dy**2)*1.5
        v_max = (acc_lat_max / abs(k).clip(1e-3))**0.5
        
        i = len(v_max)
        v_sim_acc = np.zeros(i)
        v_sim_dec = np.zeros(i)
        
        for i in range(-800,i):  #- index to simulate running start....
        
            ## max possible speed accelerating out of corners
            if v_sim_acc[i-1] < v_max[i-1]:
                acc_lat = v_sim_acc[i-1]**2 * k[i-1]
                acc_lon = acc_lon_max * (acc_lat_max**2 - acc_lat**2)**0.5 / acc_lat_max - ( v_sim_acc[i-1]**2 /1240 )
                v_sim_acc[i] =  min( (v_sim_acc[i-1]**2 + 2*acc_lon * self.s[i])**0.5 ,  v_max[i])
            else:
                #acc_lon = 0
                v_sim_acc[i] =  min( v_sim_acc[i-1] ,  v_max[i])       
    
            ## max possible speed braking into corners  (backwards lap)
            if v_sim_dec[i-1] < v_max[::-1][i-1]:
                acc_lat = v_sim_dec[i-1]**2 * k[::-1][i-1]
                acc_lon = acc_lon_min * (acc_lat_max**2 - acc_lat**2)**0.5 / acc_lat_max + ( v_sim_dec[i-1]**2 /1240 )
                v_sim_dec[i] =   min( (v_sim_dec[i-1]**2 + 2*acc_lon * self.s[::-1][i])**0.5 ,  v_max[::-1][i])
            else:
                #acc_lon = 0
                v_sim_dec[i] =  min( v_sim_dec[i-1] ,  v_max[::-1][i])

    
        v_sim_dec = v_sim_dec[::-1] #flip te matrix
        v_sim = np.minimum(v_sim_acc, v_sim_dec)
        self.laptime = s[-1] / v_sim.mean()
        return self.laptime
        
#        df['Raceline_x'] = raceline.x
#        df['Raceline_y'] = raceline.y
##        df['Raceline_z'] = df.Left_z + (df.Right_z - df.Left_z) * position
#        df['Path_S'] = S
##        df['dA'] = dA
#        df['Curvature'] = Curvature
#        df['v_max_m/s'] = v_max
#        df['v_sim_acc'] = v_sim_acc
#        df['v_sim_dec'] = v_sim_dec
#        df['v_sim'] = v_sim
#        df['a_lat'] = v_sim ** 2 * Curvature
#        df['a_lon'] = np.diff(v_sim ** 2, prepend=v_sim[-1]**2) / 2 / S
    
        
    
    def plot(self):
        plt.plot(self.x,self.y)






#    position = savgol_filter(position-df.Path_position, 11, 5, mode='wrap')+df.Path_position # window size 51, polynomial order 2

    
#    Raceline_x = df.Left_x.values + (df.Right_x.values - df.Left_x.values) * position
#    Raceline_y = df.Left_y.values + (df.Right_y.values - df.Left_y.values) * position

#    cs = CubicSpline(df.index, np.c_[Raceline_x, Raceline_y], bc_type='periodic')


#    if True: #smoothing
#        Raceline_x = savgol_filter(Raceline_x, 41, 5, mode='wrap') # window size 51, polynomial order 2
#        Raceline_y = savgol_filter(Raceline_y, 41, 5, mode='wrap') # window size 51, polynomial order 2
 
#    Raceline_x[-1] = Raceline_x[0]  #REPLACE last value with start value
#    Raceline_y[-1] = Raceline_y[0]  #REPLACE last value with start value





    
def new_Raceline(position, x):
    [mean, stdev, mag] = x
    x = np.arange(len(position))+1
    x05 = len(position)/2
    y = norm.pdf(np.roll(x,int(mean - x05)),  x05 , stdev) * stdev * mag
    position = (position + y).clip(0.05,0.95)
    return position



if __name__ == '__main__':

    acc_lat_max = 1.15 * 9.81
    acc_lon_max = 0.35 * 9.81
    acc_lon_min = 0.5  * 9.81
    
    # read starting positions from file
    df = pandas.read_csv('WP_POS.csv', index_col='Row Labels')
    
    try:
        r1 = Raceline(df.Raceline_x.values, df.Raceline_y.values)
    except:
        r1 = Raceline(df.Path_x.values, df.Path_y.values)
#    r1.reparametrize()
    s = r1.simulate()
    print('Simulated laptime = {:02.0f}:{:05.02f}'.format(s%3600//60, s%60))

    
##    LapTimeSim1(df, df.Path_position.values.clip(0.4,0.6), full_results=True)
#    first_raceline = Raceline(df.Raceline_x.values, df.Raceline_y.values)
#    LapTimeSim1(first_raceline, full_results=True)
#
#
#
#    s = LapTimeSim1(df)
#    print('Simulated laptime = {:02.0f}:{:05.02f}'.format(s%3600//60, s%60))
#    
#    retry = 0
# =============================================================================
#      
# try:
#     while True:
# 
#         x0 = np.zeros(3)
#         x0[0] = np.random.uniform(0, len(df.index))
# #        x0[1] = np.random.uniform(0, 20)
#         x0[1] = np.random.beta(2,20)*500
#         x0[2] = np.random.randn()/20
#         
#         s1 = LapTimeSim1(df, new_Raceline(df.Race_position.values.copy(), x0))
#         
#         if s-s1> 1e-3:
#             s = s1
#             
#             df.Race_position = new_Raceline(df.Race_position.values, x0) # save race_position in df
#             LapTimeSim1(df, full_results=True)
#             print('Optimized laptime = {:02.0f}:{:05.02f} - {} - {} runs'.format(s%3600//60, s%60, x0, retry))
#             retry = 0
# #            
#         else:
#             retry+=1
# 
# #        LapTimeSim1(df, full_results=True)  #saves results in df
# #        s = LapTimeSim1(df) 
# #        print('Optimized laptime = {:02.0f}:{:05.02f}'.format(s%3600//60, s%60))
# 
# 
# except KeyboardInterrupt:
#     s = LapTimeSim1(df) #save df  
#     print('Simulated laptime = {:02.0f}:{:05.02f}'.format(s%3600//60, s%60))
# 
# 
# 
# #%% save df            
#     LapTimeSim1(df, full_results=True)
#     df.to_csv('WP_POS_simulated_v4.csv')
#     
# =============================================================================
# =============================================================================
#     
#     
# #%% plot results    
# 
# 
# #    i=find_peaks(df.v_sim, width=25)[0]
#     i_acc = find_peaks(-df.v_sim, prominence=1)[0]
#     i_dec = find_peaks(df.v_sim, prominence=1)[0]
#     
#     plt.figure()
#     plt.axis('equal')
#     plt.plot(df.Left_x, df.Left_y)    
#     plt.plot(df.Right_x, df.Right_y)
#     plt.plot(df.Raceline_x, df.Raceline_y)
#     plt.plot(df.Raceline_x[i_acc], df.Raceline_y[i_acc], 'bo')
#     plt.plot(df.Raceline_x[i_dec], df.Raceline_y[i_dec], 'ro')
#     for [x,y,i] in zip(df.Raceline_x[i_acc], df.Raceline_y[i_acc], df.v_sim[i_acc]):
#         plt.annotate('{:05.02f}'.format(i*3.6), xy=[x,y], fontsize=9)
#     for [x,y,i] in zip(df.Raceline_x[i_dec], df.Raceline_y[i_dec], df.v_sim[i_dec]):
#         plt.annotate('{:05.02f}'.format(i*3.6), xy=[x,y], fontsize=9)
#     
#     
#     plt.figure()
# #    plt.plot(df['v_max_m/s']*3.6)
# #    plt.plot(df.v_sim_acc*3.6)
# #    plt.plot(df.v_sim_dec*3.6)
#     plt.plot(df.v_sim*3.6)
#     plt.plot(df.Path_distance[i_acc], df.v_sim[i_acc]*3.6, 'bo')
#     plt.plot(df.Path_distance[i_dec], df.v_sim[i_dec]*3.6, 'ro')
# #    plt.plot(df.Path_position)
#     for [x,y] in zip(df.Path_distance[i_acc], df.v_sim[i_acc]*3.6):
#         plt.annotate('{:05.02f}'.format(y), xy=[x,y], fontsize=9)
#     for [x,y] in zip(df.Path_distance[i_dec], df.v_sim[i_dec]*3.6):
#         plt.annotate('{:05.02f}'.format(y), xy=[x,y], fontsize=9)
# 
# 
# # =============================================================================
# #     plt.figure()
# #     plt.plot(df.a_lat/9.81, df.a_lon/9.81, 'ro')
# # 
# #     plt.figure()
# #     plt.plot(df.a_lon/9.81, df.v_sim*3.6, 'ro')
# # =============================================================================
# #%% plot 2
#     plt.figure()
#     plt.plot(df.Raceline_x, df.Raceline_y)
#     
# 
# =============================================================================
