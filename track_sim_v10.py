# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np 
import pandas
 

#Racechrono csv.v2 Headers
str_spd  = 'Speed (m/s)'
str_dist = 'Distance (m)'
str_time = 'Time (s)'
str_lat = 'Lateral acceleration (m/s2)'
str_lon = 'Longitudinal acceleration (m/s2)'

g = 9.81

mag = lambda v: np.sum(v**2, 1)**0.5
dot = lambda u, v: np.einsum('ij,ij->i',u,v)
#c = lambda v:v[:,None]

#import power_curve as rfs

class Track:
    def __init__(self, outside, inside, path=None):
        self.outside = outside
        self.inside = inside
        return 


class Car(dict):
    acc_limit = 1
    dec_limit = 1
    def __init__(self, *args, **kwargs):
        super(Car, self).__init__(*args, **kwargs)
        self.__dict__ = self

#    @profile
    def get_max_acc(self,v, acc_lat):
        '''maximum possible acceleration (flooring)'''

        acc_lon_max = self.acc_limit / self.acc_grip_max * (self.acc_grip_max**2 - acc_lat**2)**0.5   #grip circle (no downforce accounted for)


        
        try:
            F = force_engine(v)
        except:
            F = self.P_engine / v   #tractive force (limited by engine power)
        
        R_a =  v**2 * self.c_drag                                #aerodynamic drag
        # F = m*a + R_a + R_rl + Rg
        acc_lon = (F - R_a ) / self.mass                        
        acc_lon -=  self.c_roll * g                               #rolling resistance

        return min(acc_lon_max, acc_lon)

#    @profile        
    def get_min_acc(self,v, acc_lat):
        '''maximum possible deceleration (braking)'''

        acc_lon =  self.dec_limit * (1 - (acc_lat / self.acc_grip_max)**2)**0.5 #grip circle (no downforce accounted for)

#        if acc_lon!=acc_lon1:
#            pass

        acc_lon +=  v**2 * self.c_drag / self.mass
        acc_lon +=  self.c_roll * g #rolling resistance
        return acc_lon
        

class Raceline:
    
    def __init__(self,track, position):
        
        self.xyz = track.outside + (track.inside - track.outside) * position[:,None]  #first and last should be same point
        dz    = track.inside[:,2] - track.outside[:,2]
        width = np.sum((track.inside[:,:2] - track.outside[:,:2])**2, 1) ** 0.5
        self.ds = mag(np.diff(self.xyz.T,1 ,prepend=np.c_[self.xyz[-1]]).T)     #distance from previous
        self.s = self.ds.cumsum()                                               #station 0=start/finish
#        self.ds = np.roll(self.ds, -1)                                          #distance to next point
        self.slope = dz / width
#        self.xyz = xyz
        return


#    @profile
    def simulate(self, car):

        # Calculate the first and second derivative of the points
        dX = np.gradient(self.xyz, axis=0)
        ddX = np.gradient(dX, axis=0)

        k = mag(np.cross(dX, ddX))/mag(dX)**3# magnitude of curvature
        
        T = dX / mag(dX)[:,None]      #unit tangent (direction of travel)
        B = np.cross(dX, ddX)   #binormal
        B = B / mag(B)[:,None]          #unit binormal
        N = np.cross(B, T)      #unit normal vector
        Nk = N * k[:,None]# direction of curvature  (normal vector with magnitude 1/R)
        Tt = T# car and track share tangent vector. We're not flying
        
        #Rotate Tt 90deg CW in xy-plane
        Bt = Tt[:,[1, 0, 2]]
        Bt[:,1] *= -1        
        Bt[:,2] = self.slope         #align Bt with the track and normalize
        Bt = Bt / mag(Bt)[:,None]
        Nt = np.cross(Bt, Tt)
        
        proj_car_axis = lambda v: np.c_[dot(v, Tt), dot(v, Bt), dot(v, Nt)]
        k_car = proj_car_axis(Nk)          #curvature projected in car axis [lon, lat, z]
        g_car = proj_car_axis(np.array([0, 0, g])[None,:])   #direction of gravity in car axis [lon, lat, z]
        v_max = ((car.acc_grip_max - g_car[:,1]) / abs(k_car[:,1]).clip(1e-3))**0.5
        
        i = len(v_max)
        v_a = np.zeros(i)+1  #simulated speed maximum acceleration (+1 to avoid devision by zero)
        v_b = np.zeros(i)+1  #simulated speed maximum braking
        
      
        for i in range(-800,i):  #negative index to simulate running start....
            j = i-1 #index to previous timestep
            ## max possible speed accelerating out of corners
            v0 = v_a[j]
            if v0 < v_max[j]:     #check if previous speed was lower than max
                acc_lat = v0**2 * k_car[j,1] + g_car[j,1]                  #calc lateral acceleration based on
                acc_lon = car.get_max_acc(v0, acc_lat)
                acc_lon -= g_car[j,0]
                v1 =  (v0**2 + 2*acc_lon * self.ds[i])**0.5
                v_a[i] = min( v1 ,  v_max[i])
            else:                   #if corner speed was maximal, all grip is used for lateral acceleration (=cornering)
                acc_lon = 0         #no grip available for longitudinal acceleration
                v_a[i] = min( v0 ,  v_max[i])  #speed remains the same
                

            
            ## max possible speed braking into corners (backwards lap)
            if v_b[j] < v_max[-i]:
                acc_lat = v_b[j]**2 * k_car[-i,1] + g_car[-i,1] 
                acc_lon = car.get_min_acc(v_b[j], acc_lat)
                acc_lon += g_car[j,0]
                v =  (v_b[j]**2 + 2*acc_lon * self.ds[::-1][i])**0.5
                v_b[i] =   min(v  ,  v_max[::-1][i])
                
            else:
                acc_lon = 0
                v_b[i] =  min( v_b[j] ,  v_max[::-1][i])
                
            
        
        v_b = v_b[::-1] #flip te matrix
        self.speed = np.fmin(v_a, v_b)

        self.dt = 2 *  self.ds / (self.speed + np.roll(self.speed,1) ) 
        self.t = self.dt.cumsum()

        self.a_lat = self.speed**2 * Nk[:,1]
        self.a_lon = np.gradient(self.speed, self.s)*self.speed

        self.Bt = Bt
        self.g_car = g_car
        

        self.laptime = self.t[-1]
        return self.laptime
    
    def return_dataframe(self, df):
        df['Distance (m)']=self.s
        df['Speed (m/s)'] = self.speed
        df['Longitudinal acceleration (m/s2)'] = self.a_lon
        df['Lateral acceleration (m/s2)'] = self.a_lat
        return df

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
    import matplotlib.pyplot as plt

    ##ggv diagram /performance envelope parameters
    
# =============================================================================
#     # Peugeot 205 GTi RFS
#     race_car = Car(dict(
#             name = 'Peugeot 205 Gti RFS', 
#             acc_grip_max = 1.35 * g,
#             acc_limit = 0.4455 * g, 
#             dec_limit = 1.0125 * g, 
#             mass = 1045,
#             P_engine = 108 / 1.3410 * 1000,  #Watt  (108hp@wheels)
# #            c_drag = 0.5 * 0.34 * 1.22 * 1.78,        # ref: http://www.carinf.com/en/b41047154.html
#             c_drag = 0.5 * 0.28 * 1.21 * 1.58,        # ref: http://www.carinf.com/en/b41047154.html
#             c_roll = 0.016  #approximation (low)
#             ))
#     from power_curve_205 import force_engine, gear
#     df2 = pandas.read_csv('./logged_data/JJ_205RFS/session_zandvoort_circuit_20130604_1504_v2.csv', skiprows = 10) #205 GTi RFS
#     fastest_lap = 50
# #    df2 = pandas.read_csv('./logged_data/JJ_205RFS/session_zandvoort_circuit_20140718_2310_v2.csv', skiprows = 10) #205 GTi RFS
# =============================================================================
    
    # BMW Z3M Viperwizard
    race_car = Car(dict(
            name = 'BMW Z3M Viperwizard',
            acc_grip_max = 1.35 * g,
            acc_limit = 0.33 * 1.35 * g, 
            dec_limit = 0.60 * 1.35 * g, 
            mass = 1450,
            P_engine = 218 / 1.3410 * 1000,             #Watt  (228hp@wheels)
            c_drag = 0.5 * 0.37 * 1.22 * 1.83,        # ref:http://www.carinf.com/en/ff3031158.html
            c_roll = 0.015, #approximation
            ))
    df2 = pandas.read_csv('./logged_data/NO_BMWZ3M/session_zandvoort_circuit_20190930_2045_v2.csv', skiprows = 10) # BMW Z3M
    fastest_lap = 3

    
    # read starting positions from file
    df = pandas.read_csv('./tracks/20191004_Circuit_Zandvoort.csv')

    track_zandvoort = Track(np.c_[df.outer_x.values, df.outer_y.values, df.outer_z.values],np.c_[df.inner_x.values, df.inner_y.values, df.inner_z.values])
    try:
        raceline_csv = Raceline(track_zandvoort, df.initial_position.values + df.raceline_optimization.values)
    except AttributeError:
        raceline_csv = Raceline(track_zandvoort, df.initial_position.values)

    laptime = raceline_csv.simulate(race_car)
    print('Simulated laptime = {:02.0f}:{:05.02f} - {}'.format(laptime%3600//60, laptime%60, race_car.name))

#%% save results
    df = raceline_csv.return_dataframe(df)
    df.to_csv(race_car.name+'_Zandvoort_simulated.csv', index = None, header=True)

#%% plot  track 
    xyz = raceline_csv.xyz

    fig = plt.figure()
    plt.axis('equal')
    plt.plot(xyz[:,0],xyz[:,1])
    plt.plot(track_zandvoort.outside.T[0], track_zandvoort.outside.T[1],'r')
    plt.plot(track_zandvoort.inside.T[0], track_zandvoort.inside.T[1],'g')
    
    
    
    
    q = raceline_csv.Bt.T * raceline_csv.g_car[:,0]  #r.g_car.T
    plt.quiver(xyz[:,0],xyz[:,1], q[0], q[1])
    i = np.searchsorted(raceline_csv.s,3775, side='right')
    plt.plot(xyz[i,0],xyz[i,1], 'o')


#%%    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1_2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    df2 = df2[df2['Lap #']>0]
    # Lateral/longitudinal GG-diagram
    ax2.plot(df2[str_lat], df2[str_lon])
    ax2.plot(df[str_lat], df[str_lon])
    ax2.set_xlabel('Lateral acceleration [m/s²]')
    ax2.set_ylabel('Longitudinal acceleration [m/s²]')

    # Longitudinal gv-diagram
    ax3.plot(df2[str_lon], df2[str_spd])
    ax3.plot(df[str_lon], df[str_spd])
    ax3.set_ylabel('Velocity [m/s]')

    # Lateral gv-diagram
    ax4.plot(-df2[str_lat], df2[str_spd])
    ax4.plot(df[str_lat], df[str_spd])
            
    
    df2 = df2[df2['Lap #']==fastest_lap]
    df2[str_dist] -= df2[str_dist].values[0]
    df2[str_dist] *= df[str_dist].iloc[-1] / df2[str_dist].iloc[-1]
    df2[str_time] -= df2[str_time].values[0]
        
    t1 = raceline_csv.t
    t2 = df2[str_time].values

#    t2 = np.interp(df[str_dist], df2[str_dist], t2)
    t1 = np.interp(df2[str_dist], df[str_dist], t1)

    #plot delta t on secondary axis
    ax1_2.fill_between(df2[str_dist],0, np.gradient(t2-t1), alpha = 0.5 )

   #plot speed
    ax1.plot(df[str_dist], df[str_spd])
    ax1.plot(df2[str_dist], df2[str_spd])

#    ax1.plot(df[str_dist], gear(df[str_spd]))

