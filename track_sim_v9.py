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


g = 9.81
mag = lambda v: np.sum(v**2, 1)**0.5
dot = lambda u, v: np.einsum('ij,ij->i',u,v)
#c = lambda v:v[:,None]

class Track:

    def __init__(self, outside, inside, path=None):
        
        coor = np.r_[outside, inside][:,:2]
        elev = np.r_[outside, inside][:,2]
       
        self.outside = outside
        self.inside = inside
#        self.get_elev = LinearNDInterpolator(coor, elev)
        
        return 


class Raceline:
    
#    def __init__(self,x, y, z=None):
    def __init__(self,track, position):
        
        xyz = track.outside + (track.inside - track.outside) * position[:,None]
                
        width = np.sum((track.inside[:,:2] - track.outside[:,:2])**2, 1) ** 0.5
        dz    = track.inside[:,2] - track.outside[:,2]
        
        slope = dz / width
        
        self.slope = slope
        self.xyz = xyz
        return

    
#    def get_elev(self, track):
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
              
        
        s = mag(dX)             #distance between points
        # magnitude of curvature
        k = mag(np.cross(dX, ddX))/mag(dX)**3
        
        T = dX / s[:,None]      #unit tangent (direction of travel)
        B = np.cross(dX, ddX)   #binormal
        B = B / mag(B)[:,None]          #unit binormal
        N = np.cross(B, T)      #unit normal vector
                
        # direction of curvature  (normal vector with magnitude 1/R)
        Nk = N * k[:,None]
        
        # car and track share tangent vector. We're not flying
        Tt = T
        
        #Rotate Tt 90deg CW in xy-plane
        Bt = Tt[:,[1, 0, 2]]
        Bt[:,1] *= -1        
        Bt[:,2] = self.slope         #align Bt with the track and normalize
        Bt = Bt / mag(Bt)[:,None]
        
        Nt = np.cross(Bt, Tt)
        
        proj_car_axis = lambda v: np.c_[dot(v, Tt), dot(v, Bt), dot(v, Nt)]
        k_car = proj_car_axis(Nk)          #curvature projected in car axis [lon, lat, z]
        g_car = proj_car_axis(np.array([0, 0, g])[None,:])   #direction of gravity in car axis [lon, lat, z]
        
     
#        k = kt[:,1]
#        g_lat = gt[:,1]
#       
        v_max = ((acc_grip_max - g_car[:,1]) / abs(k_car[:,1]).clip(1e-3))**0.5
        
        i = len(v_max)
        v_a = np.zeros(i)  #simulated speed maximum acceleration
        v_b = np.zeros(i)  #simulated speed maximum braking
        
      
        for i in range(-800,i):  #negative index to simulate running start....
            j = i-1 #index to previous timestep
            ## max possible speed accelerating out of corners
            if v_a[j] < v_max[j]:     #check if previous speed was lower than max

                acc_lat = v_a[j]**2 * k_car[j,1] + g_car[j,1]                  #calc lateral acceleration based on
                acc_lon = acc_lon_max / acc_grip_max * (acc_grip_max**2 - acc_lat**2)**0.5   #grip circle (no downforce accounted for)
                acc_lon -=  v_a[j]**2 * c_drag                                 #aerodynamic drag + 
                acc_lon -=  v_a[j] * c_roll                                    #rolling resistance + 
                if v_a[j]>27:
                    print()
                
                acc_lon -=  g_car[j,0]                                         #gravity up/down hill
                v_a[i] =  min( (v_a[j]**2 + 2*acc_lon * s[j])**0.5 ,  v_max[i])
            else:
                #acc_lon = 0
                v_a[i] =  min( v_a[j] ,  v_max[i])

            ## max possible speed braking into corners  (backwards lap)
            if v_b[j] < v_max[-i]:
                acc_lat = v_b[j]**2 * k_car[-i,1] + g_car[-i,1] 
                acc_lon = acc_lon_min  / acc_grip_max * (acc_grip_max**2 - acc_lat**2)**0.5
                acc_lon +=  v_b[j]**2 * c_drag
                acc_lon +=  v_b[j] * c_roll 
#                acc_lon +=  g_car[j,0]
                v_b[i] =   min( (v_b[j]**2 + 2*acc_lon * s[::-1][j])**0.5 ,  v_max[::-1][i])
            else:
                #acc_lon = 0
                v_b[i] =  min( v_b[j] ,  v_max[::-1][i])
    
    
        v_b = v_b[::-1] #flip te matrix
        self.speed = np.fmin(v_a, v_b)

        self.a_lat = self.speed**2 * Nk[:,1]
        self.a_lon = np.gradient(self.speed, s.cumsum())*self.speed

#        self.Nk = Nk
#        self.Nt = Nt
        self.Bt = Bt
        
        self.s = s
        

        self.laptime = sum(s) / self.speed.mean()
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
    
    # Peugeot 205 GTi RFS
    acc_grip_max = 1.15 * 9.81
    #0.5 * C_car * rho_air * Area_car / Mass_car (C is car sepecific drag cooficient)  a=F/m
    m_car = 1050
    c_drag = 0.5 * 0.34 * 1.22 * 1.78        # ref: http://www.carinf.com/en/b41047154.html
    c_roll = 0.00  #approximation (low)
    v_max = 235 / 3.6
    acc_lon_max = (v_max**2 * c_drag + v_max * c_roll  )/m_car
    acc_lon_min = acc_grip_max
    
    
# =============================================================================
#     # BMW Z3M Viperwizard
#     acc_grip_max = 1.35 * 9.81
#     m_car = 950
# #    c_drag = 0.5 * 0.37 * 1.22 * 1.83        # ref: http://www.carinf.com/en/b41047154.html
#     c_drag = 0.5 * 0.37 * 1.22 * 1.83        # ref: http://www.carinf.com/en/b41047154.html
#     c_roll = 0.015 #approximation
#     acc_lon_max = 0.45 * 9.81
#     acc_lon_min = acc_grip_max * 0.5
# =============================================================================


    c_drag /= m_car
    c_roll /= m_car

    
    # read starting positions from file
    df = pandas.read_csv('./tracks/20191004_Circuit_Zandvoort.csv', index_col='Row Labels')

    track_zandvoort = Track(np.c_[df.outer_x.values, df.outer_y.values, df.outer_z.values],np.c_[df.inner_x.values, df.inner_y.values, df.inner_z.values])
    try:
        raceline_csv = Raceline(track_zandvoort, df.initial_position.values + df.raceline_optimization.values)
    except AttributeError:
        raceline_csv = Raceline(track_zandvoort, df.initial_position.values)
            
        

    s = raceline_csv.simulate(track_zandvoort)
    print('Simulated laptime = {:02.0f}:{:05.02f}'.format(s%3600//60, s%60))
    
        
##%% plot    
    r=raceline_csv
    x, y, z = r.xyz[:,0], r.xyz[:,1], r.xyz[:,2]

    fig = plt.figure()
    
    plt.axis('equal')
#    fig.add_subplot(111, projection='3d')
    

    
    plt.plot(x,y)
    plt.plot(track_zandvoort.outside.T[0], track_zandvoort.outside.T[1],'r')
    plt.plot(track_zandvoort.inside.T[0], track_zandvoort.inside.T[1],'g')
#    plt.plot(df.Right_x,df.Right_y, 'g')
    
    q = r.Bt.T * r.speed  #r.g_car.T
    plt.quiver(x,y, q[0], q[1])
#    plt.quiver(x,y, raceline_csv.N[:,0], raceline_csv.N[:,1], scale_units =1)

#    plt.figure()
#    plt.plot(r.speed*3.6)

    
    df['Distance (m)']=r.s.cumsum()
    df['Speed (kph)'] = r.speed*3.6
    df['Longitudinal Acceleration (G)'] = r.a_lon/9.81
    df['Lateral Acceleration (G)'] = r.a_lat/9.81
#    
    df.to_csv('Z3M_Zandvoort_simulated.csv', index = None, header=True)
    
    import track_compare
    