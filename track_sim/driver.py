# from sim import Car, Track
from typing import Any
import numpy as np

def mag(vector):
    return np.sum(vector**2, 1)**0.5


def dot(u, v):
    return np.einsum('ij,ij->i',u,v)


def get_curvature(line):
    dX = np.gradient(line, axis=0)
    ddX = np.gradient(dX, axis=0)
    return mag(np.cross(dX, ddX))/mag(dX)**3 


class Driver():
    def __init__(self, car: Any, track: Any, raceline: Any = None):
        self.car = car
        self.track = track
        self.position_clearance = track.min_clearance / track.width
                
        if raceline is None:
            raceline = np.ones(np.size(self.position_clearance))/2

        self.raceline = raceline
        self.pr = self.race(raceline)


    def try_new_line(self):
        start = np.random.randint(0, len(self.raceline))
        length = np.random.randint(1, 60)
        deviation = np.random.randn() / 10
       
        line_adjust = 1 - np.cos(np.linspace(0, 2*np.pi, length))
        new_line = self.raceline * 0
        new_line[:length] = line_adjust * deviation
        new_line = np.roll(new_line, start)
        test_line = self.raceline + new_line / self.track.width
        test_line  = np.clip(test_line, self.position_clearance, 1 - self.position_clearance)

        laptime = self.race(test_line)

        if laptime < self.pr:
            self.pr = laptime
            self.raceline = test_line
            
        return self.pr


    def race(self, raceline):
     
        line = self.track.get_line_coordinates(raceline)
        ds = mag(np.diff(line.T,1 ,prepend=np.c_[line[-1]]).T)     #distance from previous

        # Calculate the first and second derivative of the points
        dX = np.gradient(line, axis=0)
        ddX = np.gradient(dX, axis=0)

        k = mag(np.cross(dX, ddX))/mag(dX)**3   # magnitude of curvature

        T = dX / mag(dX)[:,None]        # unit tangent (direction of travel)
        B = np.cross(dX, ddX)           # binormal
        B = B / mag(B)[:,None]          # unit binormal
        N = np.cross(B, T)              # unit normal vector
        Nk = N * k[:,None]              # direction of curvature  (normal vector with magnitude 1/R)
        Tt = T                          # car and track share tangent vector. We're not flying

        #Rotate Tt 90deg CW in xy-plane
        Bt = Tt[:,[1, 0, 2]]
        Bt[:,1] *= -1
        Bt[:,2] = self.track.slope         #align Bt with the track and normalize
        Bt = Bt / mag(Bt)[:,None]
        Nt = np.cross(Bt, Tt)

        car_axis = lambda x: np.c_[dot(x, Tt), dot(x, Bt), dot(x, Nt)]
        k_car = car_axis(Nk)          #curvature projected in car frame [lon, lat, z]
        g_car = car_axis(np.array([0, 0, 9.81])[None,:])   #direction of gravity in car frame [lon, lat, z]

        k_car = k_car[:,1]
        k_car = np.sign(k_car) * np.abs(k_car).clip(1e-3)

        v_max = np.abs((self.car.acc_grip_max - np.sign(k_car) * g_car[:,1]) / k_car)**0.5

        v_a = np.zeros(len(v_max))+1  #simulated speed maximum acceleration (+1 to avoid division by zero)
        v_b = np.zeros(len(v_max))+1  #simulated speed maximum braking

        for i in range(-100,len(v_max)):  #negative index to simulate running start....
            j = i-1 #index to previous timestep

            ## max possible speed accelerating out of corners
            if v_a[j] < v_max[j]:     #check if previous speed was lower than max

                acc_lat = v_a[j]**2 * k_car[j] + g_car[j,1]                  #calc lateral acceleration based on
                acc_lon = self.car.get_max_acc(v_a[j], acc_lat)
                acc_lon -= g_car[j,0]
                v1 =  (v_a[j]**2 + 2*acc_lon * ds[i])**0.5
                v_a[i] = min( v1 ,  v_max[i])
            else:                   #if corner speed was maximal, all grip is used for lateral acceleration (=cornering)
                acc_lon = 0         #no grip available for longitudinal acceleration
                v_a[i] = min( v_a[j] ,  v_max[i])  #speed remains the same


            ## max possible speed braking into corners (backwards lap)
            v0 = v_b[j]
            if v0 < v_max[::-1][j]:

                acc_lat = v0**2 * k_car[::-1][j] + g_car[::-1][j,1] 
                acc_lon = self.car.get_min_acc(v0, acc_lat)
                acc_lon += g_car[::-1][j,0]
                v1 =  (v0**2 + 2*acc_lon * ds[::-1][i])**0.5
                v_b[i] =   min(v1 ,  v_max[::-1][i])

            else:
                acc_lon = 0
                v_b[i] =  min( v0 ,  v_max[::-1][i])

        v_b = v_b[::-1] #flip te matrix
        speed = np.fmin(v_a, v_b)
        dt = 2 *  ds / (speed + np.roll(speed,1) )
        time = dt.cumsum()


        return time[-1]



    def race_results(self):

        line = self.track.get_line_coordinates(self.raceline)
        ds = mag(np.diff(line.T,1 ,prepend=np.c_[line[-1]]).T)     #distance from previous
        distance = ds.cumsum() - ds[0]

        # Calculate the first and second derivative of the points
        dX = np.gradient(line, axis=0)
        ddX = np.gradient(dX, axis=0)

        k = mag(np.cross(dX, ddX))/mag(dX)**3   # magnitude of curvature

        T = dX / mag(dX)[:,None]        # unit tangent (direction of travel)
        B = np.cross(dX, ddX)           # binormal
        B = B / mag(B)[:,None]          # unit binormal
        N = np.cross(B, T)              # unit normal vector
        Nk = N * k[:,None]              # direction of curvature  (normal vector with magnitude 1/R)
        Tt = T                          # car and track share tangent vector. We're not flying

        #Rotate Tt 90deg CW in xy-plane
        Bt = Tt[:,[1, 0, 2]]
        Bt[:,1] *= -1
        Bt[:,2] = self.track.slope         #align Bt with the track and normalize
        Bt = Bt / mag(Bt)[:,None]
        Nt = np.cross(Bt, Tt)

        car_axis = lambda x: np.c_[dot(x, Tt), dot(x, Bt), dot(x, Nt)]
        k_car = car_axis(Nk)          #curvature projected in car frame [lon, lat, z]
        g_car = car_axis(np.array([0, 0, 9.81])[None,:])   #direction of gravity in car frame [lon, lat, z]

        k_car = k_car[:,1]
        k_car = np.sign(k_car) * np.abs(k_car).clip(1e-3)

        v_max = np.abs((self.car.acc_grip_max - np.sign(k_car) * g_car[:,1]) / k_car)**0.5

        v_a = np.zeros(len(v_max))+1  #simulated speed maximum acceleration (+1 to avoid division by zero)
        v_b = np.zeros(len(v_max))+1  #simulated speed maximum braking

        for i in range(-100,len(v_max)):  #negative index to simulate running start....
            j = i-1 #index to previous timestep

            ## max possible speed accelerating out of corners
            if v_a[j] < v_max[j]:     #check if previous speed was lower than max

                acc_lat = v_a[j]**2 * k_car[j] + g_car[j,1]                  #calc lateral acceleration based on
                acc_lon = self.car.get_max_acc(v_a[j], acc_lat)
                acc_lon -= g_car[j,0]
                v1 =  (v_a[j]**2 + 2*acc_lon * ds[i])**0.5
                v_a[i] = min( v1 ,  v_max[i])
            else:                   #if corner speed was maximal, all grip is used for lateral acceleration (=cornering)
                acc_lon = 0         #no grip available for longitudinal acceleration
                v_a[i] = min( v_a[j] ,  v_max[i])  #speed remains the same


            ## max possible speed braking into corners (backwards lap)
            v0 = v_b[j]
            if v0 < v_max[::-1][j]:

                acc_lat = v0**2 * k_car[::-1][j] + g_car[::-1][j,1] 
                acc_lon = self.car.get_min_acc(v0, acc_lat)
                acc_lon += g_car[::-1][j,0]
                v1 =  (v0**2 + 2*acc_lon * ds[::-1][i])**0.5
                v_b[i] =   min(v1 ,  v_max[::-1][i])

            else:
                acc_lon = 0
                v_b[i] =  min( v0 ,  v_max[::-1][i])

        v_b = v_b[::-1] #flip te matrix
        speed = np.fmin(v_a, v_b)
        dt = 2 *  ds / (speed + np.roll(speed,1) )
        time = dt.cumsum()

        a_lat = -(speed**2) * Nk[:, 0]
        a_lon = np.gradient(speed, distance) * speed

        return np.column_stack((self.raceline, distance, line, speed, time, a_lat, a_lon))
