
import numpy as np 
import pandas as pd
import json

from track_sim.track import Track
from track_sim.car import Car
from utilities.dotdict import DotDict



gravity = 9.81

FILENAME_CAR_PROPERTIES = './cars/BMW_Z3M.json'
# filename_log = './session_zandvoort_circuit_20190930_2045_v2.csv'
FILENAME_TRACK = './tracks/20191030_Circuit_Zandvoort copy.csv'

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

 
def laptime_str(seconds):
    return "{:02.0f}:{:06.03f}".format(seconds%3600//60, seconds%60)


def simulate(car: Car, track: Track, position):

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
    except FileNotFoundError:
        df_track = pd.read_csv(FILENAME_TRACK)

    track = Track(
        df_track[['outer_x','outer_y','outer_z']].values,
        df_track[['inner_x','inner_y','inner_z']].values,
        )

    if 'Optimized line' in df_track.columns:
        race_line = df_track['Optimized line'].values
    elif 'initial_position' in df_track.columns:
        race_line = df_track['initial_position'].values
    else:
        df_track['Optimized line'] = 0.5
        race_line = df_track['Optimized line']



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