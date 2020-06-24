# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas
import numpy as np 
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.misc import derivative
# =============================================================================
# class Point:
#     def __init__(self, x, y, z=None):
#         self.x = x
#         self.y = y
#         self.z = z
#         return
# =============================================================================
    

class Track:

    def __init__(self, left, right, path=None):
        
        coor = np.r_[left, right][:,:2]
        elev = np.r_[left, right][:,2]
       
        self.left = left
        self.right = right
        self.get_elev = LinearNDInterpolator(coor, elev)

        return 

    def get_normals(self, x,y):
        
        dx = derivative(self.get_elev, x, dx=0.1, n=1)
        dy = derivative(self.get_elev, y, dx=0.1, n=1)
            
        
        return np.c_[dx,dy]



class Raceline:
    
    def __init__(self,x, y, z=None):

        self.x=x
        self.y=y
        self.z=z        
        return

    
    def set_elev(self, track):
        self.z = track.get_elev(np.c_[self.x, self.y])
        return self

    
    def simulate(self):
        # Calculate the first and second derivative of the points
        points = np.c_[self.x, self.y, self.z]
        
        dX = np.apply_along_axis(np.gradient, axis=0, arr=points)
        ddX = np.apply_along_axis(np.gradient, axis=0, arr=dX)

# =============================================================================
#         # Normalize all tangents 
#         f = lambda m : m / np.linalg.norm(m)
#         T = np.apply_along_axis(f, axis=1, arr=dX)
# =============================================================================
        	
        # Calculate and normalize all binormals
        B = np.cross(dX, ddX)
        k = np.cross(B, dX)         #onyl works if X spaced evenly (=constant velocity)
#        B = np.apply_along_axis(f, axis=1, arr=B)

        # Calculate all normals
#        N = np.cross(B, T)
        self.k = k

#        self.T = T
#        self.B = B
#        self.N = N
        
#        k1 = np.sum(k**2, 1)**0.5
        k = np.linalg.norm(k,None,1)
        s = np.sum(dX**2, 1)**0.5
        
        v_max = (acc_lat_max / k.clip(1e-3))**0.5
        
        i = len(v_max)
        v_sim_acc = np.zeros(i)  #simulated speed maximum acceleration
        v_sim_dec = np.zeros(i)  #simulated speed maxumum deceleration

        for i in range(-800,i):  #negative index to simulate running start....

            ## max possible speed accelerating out of corners
            if v_sim_acc[i-1] < v_max[i-1]:     #check if previous speed was lower than max
                acc_lat = v_sim_acc[i-1]**2 * k[i-1]        #calc lateral acceleration based on
                acc_lon = acc_lon_max * (acc_lat_max**2 - acc_lat**2)**0.5 / acc_lat_max  #grip circle (no downforce accounted for)
                acc_lon -= ( v_sim_acc[i-1]**2 * drag_coof )                          #aerodynamic drag
                v_sim_acc[i] =  min( (v_sim_acc[i-1]**2 + 2*acc_lon * s[i])**0.5 ,  v_max[i])
            else:
                #acc_lon = 0
                v_sim_acc[i] =  min( v_sim_acc[i-1] ,  v_max[i])

            ## max possible speed braking into corners  (backwards lap)
            if v_sim_dec[i-1] < v_max[::-1][i-1]:
                acc_lat = v_sim_dec[i-1]**2 * k[::-1][i-1]
                acc_lon = acc_lon_min * (acc_lat_max**2 - acc_lat**2)**0.5 / acc_lat_max 
                acc_lon += ( v_sim_dec[i-1]**2 * drag_coof)
                v_sim_dec[i] =   min( (v_sim_dec[i-1]**2 + 2*acc_lon * s[::-1][i])**0.5 ,  v_max[::-1][i])
            else:
                #acc_lon = 0
                v_sim_dec[i] =  min( v_sim_dec[i-1] ,  v_max[::-1][i])
    
        v_sim_dec = v_sim_dec[::-1] #flip te matrix

        self.v_sim = np.minimum(v_sim_acc, v_sim_dec)
        self.laptime = sum(s) / self.v_sim.mean()
        return self.laptime


# =============================================================================
# def new_Raceline(position, x):
#     [mean, stdev, mag] = x
#     x = np.arange(len(position))+1
#     x05 = len(position)/2
#     y = norm.pdf(np.roll(x,int(mean - x05)),  x05 , stdev) * stdev * mag
#     position = (position + y).clip(0.05,0.95)
#     return position
# =============================================================================



if __name__ == '__main__':

    ##ggv diagram /performance envelope parameters
    acc_lat_max = 1.15 * 9.81
    acc_lon_max = 0.35 * 9.81
    acc_lon_min = acc_lat_max # diameter grip circle should be constant...
    drag_coof = 1/1240  #0.5 * C_car * rho_air * Area_car / Mass_car (C is car sepecific drag cooficient)
    
    # read starting positions from file
    df = pandas.read_csv('WP_POS_simulated_v2.csv', index_col='Row Labels')
    
    track_zandvoort = Track(np.c_[df.Left_x, df.Left_y, df.Left_z],np.c_[df.Right_x, df.Right_y, df.Right_z])
    
    
#    points = np.c_[df.Raceline_x.values, df.Raceline_y.values, df.Raceline_z.values]
#    points = np.c_[df.Raceline_x, df.Raceline_y, df.Raceline_z]

    raceline_csv = Raceline(df.Raceline_x, df.Raceline_y, df.Raceline_z)
#    raceline_csv = Raceline(df.Raceline_x, df.Raceline_y)
    raceline_csv.set_elev(track_zandvoort)

    N = track_zandvoort.get_normals(raceline_csv.x, raceline_csv.y)
    
    s = raceline_csv.simulate()
    print('Simulated laptime = {:02.0f}:{:05.02f}'.format(s%3600//60, s%60))
        
        
#%% plot    
    
    
    fig = plt.figure()
    plt.axis('equal')
#    fig.add_subplot(111, projection='3d')
#    fig.add_subplot(111, projection='3d')
#    plt.plot(points[:,0],points[:,1],points[:,2])
    plt.plot(raceline_csv.x, raceline_csv.y)
    plt.plot(df.Left_x,df.Left_y, 'r')
    plt.plot(df.Right_x,df.Right_y, 'g')
    plt.quiver(raceline_csv.x,raceline_csv.y, raceline_csv.k[:,0], raceline_csv.k[:,1])

    plt.figure()
    plt.plot(raceline_csv.v_sim*3.6)
