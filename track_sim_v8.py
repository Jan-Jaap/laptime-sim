# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.misc import derivative


g = np.array([0, 0, -9.81])
mag = lambda v: np.sum(v**2, 1)**0.5
dot = lambda u, v: np.einsum('ij,ij->i',u,v)
#c = lambda v:v[:,None]

class Track:

    def __init__(self, outside, inside, path=None):
        
        coor = np.r_[outside, inside][:,:2]
        elev = np.r_[outside, inside][:,2]
       
        self.outside = outside
        self.inside = inside
        self.get_elev = LinearNDInterpolator(coor, elev)
        
        return 


class Raceline:
    
    def __init__(self,x, y, z=None):

        self.xyz = np.c_[x, y, z]
#        self.y=y
#        self.z=z        
        return

    
#    def set_elev(self, track):
#        self.z = track.get_elev(np.c_[self.x, self.y])
#        return self
#
#    def set_normal(self, track):
#        pass
                


#    @profile
    def simulate(self, track):
        # Calculate the first and second derivative of the points
        dX = np.gradient(self.xyz, axis=0)
        ddX = np.gradient(dX, axis=0)
        
#        Nk = np.cross(np.cross(dX, ddX), dX)
        
               
        
        s = mag(dX)             #distance between points
        # magnitude of curvature
        k = mag(np.cross(dX, ddX))/mag(dX)**3
        
        T = dX / s[:,None]      #unit tangent (direction of travel)
        B = np.cross(dX, ddX)   #binormal
        B = B / mag(B)[:,None]          #unit binormal
        N = np.cross(B, T)      #unit normal vector
        
        
        # direction of curvature  (normal vector with magnitude 1/R)
        Nk = N * k[:,None]
        
        # car and track share tangent vector
        Tt = T
        
        #Rotate Tt 90deg CW in xy-plane
        Bt = Tt[:,[1, 0, 2]]*0.01
        Bt[:,1] *= -1
       
        #align Bt with the track and normalize
        Bt[:,2] = (track.get_elev(self.xyz[:,:2] + Bt[:,:2]) - self.xyz[:,2])
        Bt = Bt / mag(Bt)[:,None]
        
        Nt = np.cross(Bt, Tt)
        
        proj_on_track = lambda v: np.c_[dot(v, Tt), dot(v, Bt), dot(v, Nt)]
        kt = proj_on_track(Nk)
        gt = proj_on_track(g[None,:])
        
     
        k = kt[:,1]
        g_lat = gt[:,1]
       
        v_max = ((acc_grip_max + g_lat) / abs(k).clip(1e-3))**0.5
        
        i = len(v_max)
        v_a = np.zeros(i)  #simulated speed maximum acceleration
        v_b = np.zeros(i)  #simulated speed maximum braking

        for i in range(-800,i):  #negative index to simulate running start....
            j = i-1 #index to previous timestep
            ## max possible speed accelerating out of corners
            if v_a[j] < v_max[j]:     #check if previous speed was lower than max

                acc_lat = v_a[j]**2 * k[j] - g_lat[j]        #calc lateral acceleration based on
                acc_lon = acc_lon_max / acc_grip_max * (acc_grip_max**2 - acc_lat**2)**0.5   #grip circle (no downforce accounted for)

                acc_lon -= ( v_a[j]**2 * drag_coof )                          #aerodynamic drag
                v_a[i] =  min( (v_a[j]**2 + 2*acc_lon * s[j])**0.5 ,  v_max[i])
            else:
                #acc_lon = 0
                v_a[i] =  min( v_a[j] ,  v_max[i])

            ## max possible speed braking into corners  (backwards lap)
            if v_b[j] < v_max[-i]:
                acc_lat = v_b[j]**2 * k[-i] - g_lat[-i] 
                acc_lon = (acc_grip_max**2 - acc_lat**2)**0.5
                acc_lon += ( v_b[j]**2 * drag_coof)
                v_b[i] =   min( (v_b[j]**2 + 2*acc_lon * s[::-1][j])**0.5 ,  v_max[::-1][i])
            else:
                #acc_lon = 0
                v_b[i] =  min( v_b[j] ,  v_max[::-1][i])
    
    
        v_b = v_b[::-1] #flip te matrix
        self.v_sim = np.minimum(v_a, v_b)


        self.Nk = Nk
        self.Nt = Nt
        self.Bt = Bt
        self.k = k
        self.g_lat = g_lat
        self.v_max = v_max
        

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
    acc_grip_max = 1.15 * 9.81
    acc_lon_max = 0.25 * 9.81
    drag_coof = 1/1240  #0.5 * C_car * rho_air * Area_car / Mass_car (C is car sepecific drag cooficient)
    
    # read starting positions from file
    df = pandas.read_csv('WP_POS_simulated_v2.csv', index_col='Row Labels')
    
    track_zandvoort = Track(np.c_[df.Left_x, df.Left_y, df.Left_z],np.c_[df.Right_x, df.Right_y, df.Right_z])
    

    raceline_csv = Raceline(df.Raceline_x, df.Raceline_y, df.Raceline_z)
#    raceline_csv = Raceline(df.Raceline_x, df.Raceline_y)

    s = raceline_csv.simulate(track_zandvoort)
    print('Simulated laptime = {:02.0f}:{:05.02f}'.format(s%3600//60, s%60))
        
        
##%% plot    
    r=raceline_csv
    x, y, z = r.xyz[:,0], r.xyz[:,1], r.xyz[:,2]

    
    fig = plt.figure()
    
    plt.axis('equal')
#    fig.add_subplot(111, projection='3d')
#    fig.add_subplot(111, projection='3d')
#    plt.plot(points[:,0],points[:,1],points[:,2])

    
    plt.plot(x,y)
    plt.plot(track_zandvoort.outside.T[0], track_zandvoort.outside.T[1],'r')
    plt.plot(track_zandvoort.inside.T[0], track_zandvoort.inside.T[1],'g')
#    plt.plot(df.Right_x,df.Right_y, 'g')
    
    q = r.Bt.T * r.v_sim
    plt.quiver(x,y, q[0], q[1])
#    plt.quiver(x,y, raceline_csv.N[:,0], raceline_csv.N[:,1], scale_units =1)

    plt.figure()
    plt.plot(r.v_sim*3.6)
#
#    plt.figure()
#    i = 800
#    plt.quiver(0,0, sum(q[:2,i]**2)**0.5 , q[2,i], angles='xy', scale_units='xy', scale=1)
    