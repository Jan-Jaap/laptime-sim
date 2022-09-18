
import datetime
import numpy as np 
import pandas as pd
import json
from plotly.subplots import make_subplots
import plotly.io as pio

# import utm
import matplotlib.pyplot as plt

from dotdict import DotDict



gravity = 9.81

FILENAME_CAR_PROPERTIES = './cars/BMW_Z3M.json'
# filename_log = './session_zandvoort_circuit_20190930_2045_v2.csv'
FILENAME_TRACK = './tracks/20191030_Circuit_Zandvoort.csv'

#Racechrono csv.v2 Headers
rc_header = dict(
        speed  = 'Speed (m/s)',
        distance = 'Distance (m)',
        time = 'Time (s)',
        a_lat = 'Lateral acceleration (m/s2)',
        a_lon = 'Longitudinal acceleration (m/s2)',
        )

def mag(vector):
    return np.sum(vector**2, 1)**0.5

def dot(u, v):
    return np.einsum('ij,ij->i',u,v)

class Track:
    new_line_parameters = []
    def __init__(self, outside, inside):
        self.width = np.sum((inside[:,:2] - outside[:,:2])**2, 1) ** 0.5
        self.slope =  (inside[:,2] - outside[:,2]) / self.width
        self.outside = outside
        self.inside = inside
        return
    
    def calc_line(self, position=None):

        if position is None:
            position = self.position
        position = np.clip(position, a_min = 0.1, a_max = 0.9)
        
        self.line = self.outside + (self.inside - self.outside) * position[:,None]  
        self.ds = mag(np.diff(self.line.T,1 ,prepend=np.c_[self.line[-1]]).T)     #distance from previous
        self.s = self.ds.cumsum() - self.ds[0]
        return

    def new_line(self, position):
        start = np.random.randint(0, len(self.width))
#        start = np.random.randint(3000, 3400)
        length = np.random.randint(1, 50)
        deviation = np.random.randn() / 10

        self.new_line_parameters = dict(start=start, length=length, deviation=deviation)        
        
        line_adjust = (1 - np.cos(np.linspace(0, 2*np.pi, length))) * deviation

        new_line = self.width * 0
        new_line[:length] += line_adjust
        new_line = np.roll(new_line, start)
        new_line /= self.width

        position = position + new_line
        return position

class Car(dict):
    trail_braking = 100
    
    def __init__(self, *args, **kwargs):
        super(Car, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def get_max_acc(self,v, acc_lat):
        '''maximum possible acceleration (flooring)'''
        acc_lon_max = self.acc_limit / self.acc_grip_max * (self.acc_grip_max**2 - acc_lat**2)**0.5   #grip circle (no downforce accounted for)
        acc_lon = (self.force_engine(v) - (v**2 * self.c_drag) ) / self.mass                        
        acc_lon -=  self.c_roll * gravity                               #rolling resistance
        return min(acc_lon_max, acc_lon)

    def force_engine(self, v):
        return self.P_engine / v   #tractive force (limited by engine power)

    def get_min_acc(self,v, acc_lat):
        '''maximum possible deceleration (braking)'''
        n = self.trail_braking / 50
        acc_lon = self.dec_limit * (1 - (np.abs(acc_lat) / self.acc_grip_max)**n)**(1/n)
        acc_lon +=  v**2 * self.c_drag / self.mass
        acc_lon +=  self.c_roll * gravity #rolling resistance
        return acc_lon

    def get_gear(self, v):
        return v*0

   
def laptime_str(seconds):
    return "{:02.0f}:{:06.03f}".format(seconds%3600//60, seconds%60)


def simulate(car, track, position):

    track.calc_line(position)
    
    # Calculate the first and second derivative of the points
    dX = np.gradient(track.line, axis=0)
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
    Bt[:,2] = track.slope         #align Bt with the track and normalize
    Bt = Bt / mag(Bt)[:,None]
    Nt = np.cross(Bt, Tt)
    
    car_axis = lambda x: np.c_[dot(x, Tt), dot(x, Bt), dot(x, Nt)]
    k_car = car_axis(Nk)          #curvature projected in car frame [lon, lat, z]
    g_car = car_axis(np.array([0, 0, gravity])[None,:])   #direction of gravity in car frame [lon, lat, z]
    
    k_car = k_car[:,1]
    k_car = np.sign(k_car) * np.abs(k_car).clip(1e-3)
    
    v_max = np.abs((car.acc_grip_max - np.sign(k_car) * g_car[:,1]) / k_car)**0.5

    v_a = np.zeros(len(v_max))+1  #simulated speed maximum acceleration (+1 to avoid devision by zero)
    v_b = np.zeros(len(v_max))+1  #simulated speed maximum braking
      
    for i in range(-100,len(v_max)):  #negative index to simulate running start....
        j = i-1 #index to previous timestep

        ## max possible speed accelerating out of corners
        if v_a[j] < v_max[j]:     #check if previous speed was lower than max

            acc_lat = v_a[j]**2 * k_car[j] + g_car[j,1]                  #calc lateral acceleration based on
            acc_lon = car.get_max_acc(v_a[j], acc_lat)
            acc_lon -= g_car[j,0]
            v1 =  (v_a[j]**2 + 2*acc_lon * track.ds[i])**0.5
            v_a[i] = min( v1 ,  v_max[i])
        else:                   #if corner speed was maximal, all grip is used for lateral acceleration (=cornering)
            acc_lon = 0         #no grip available for longitudinal acceleration
            v_a[i] = min( v_a[j] ,  v_max[i])  #speed remains the same

        
        ## max possible speed braking into corners (backwards lap)
        v0 = v_b[j]
        if v0 < v_max[::-1][j]:

            acc_lat = v0**2 * k_car[::-1][j] + g_car[::-1][j,1] 
            acc_lon = car.get_min_acc(v0, acc_lat)
            acc_lon += g_car[::-1][j,0]
            v1 =  (v0**2 + 2*acc_lon * track.ds[::-1][i])**0.5
            v_b[i] =   min(v1 ,  v_max[::-1][i])
            
        else:
            acc_lon = 0
            v_b[i] =  min( v0 ,  v_max[::-1][i])
      
    
    v_b = v_b[::-1] #flip te matrix
    
    results = DotDict()
    results.speed = np.fmin(v_a, v_b)
    results.dt = 2 *  track.ds / (results.speed + np.roll(results.speed,1) ) 
    results.t = results.dt.cumsum()
    results.a_lat = -results.speed**2 * Nk[:,1]
    results.a_lon = np.gradient(results.speed, track.s)*results.speed
    results.s = track.s
    results.laptime = results.t[-1]
    results.v_max = v_max
    results.race_line = position
    results.gear = car.get_gear(results.speed)
    results.new_line_parameters = track.new_line_parameters
    return results

def return_dataframe(df, results):
    df['Distance (m)']=results.s
    df['Speed (m/s)'] = results.speed
    df['Longitudinal acceleration (m/s2)'] = results.a_lon
    df['Lateral acceleration (m/s2)'] = results.a_lat
    df['Optimized line'] = results.race_line
    df['Timestamp'] = results.t
    return df

#%% main scripts
def main():


    with open(FILENAME_CAR_PROPERTIES, 'r') as fp:
        race_car = Car(json.load(fp))

    filename_results = f'./simulated/{race_car.name}_Zandvoort_simulated.csv'
    nr_iterations = 0

    try:
        df_track = pd.read_csv(filename_results)
        race_line = df_track['Optimized line'].values
    except FileNotFoundError:
        df_track = pd.read_csv(FILENAME_TRACK)
        race_line = df_track.initial_position.values


    track = Track(np.c_[df_track.outer_x.values, df_track.outer_y.values, df_track.outer_z.values],
                            np.c_[df_track.inner_x.values, df_track.inner_y.values, df_track.inner_z.values])


    results = simulate(race_car, track, race_line)
    print(f'{race_car.name} - Simulated laptime = {laptime_str(results.laptime)}')

    optimize_yn = input('Start line optimization? [y/N]')
    
    try:
        while optimize_yn in ['y', 'Y']:
            new_race_line = track.new_line(results.race_line)
            results_temp = simulate(race_car, track, new_race_line)
            nr_iterations += 1
            laptime_improvement = results.laptime - results_temp.laptime
            if laptime_improvement > 0:
                results = results_temp

                print(f"Laptime = {laptime_str(results.laptime)}  (iteration:{nr_iterations})")

    except KeyboardInterrupt:
        print('Interrupted by CTRL+C, saving progress')


#%% save results
    df_track = return_dataframe(df_track, results)
    df_track.to_csv(filename_results, index = None, header=True)

    results.speed *= 3.6  #convert speed from m/s to km/h

    print(f'{race_car.name} - Simulated laptime = {results.laptime%3600//60:02.0f}:{results.laptime%60:05.02f}')


if __name__ == '__main__':
    main()