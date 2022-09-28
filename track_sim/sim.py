from dataclasses import dataclass

from utilities.dotdict import DotDict
import numpy as np
import pandas as pd

def mag(vector):
    return np.sum(vector**2, 1)**0.5


def dot(u, v):
    return np.einsum('ij,ij->i',u,v)


class Car(DotDict):
    trail_braking = 70
    
    def get_max_acc(self,v, acc_lat):
        '''maximum possible acceleration (flooring)'''
        acc_lon_max = self.acc_limit / self.acc_grip_max * (self.acc_grip_max**2 - acc_lat**2)**0.5   #grip circle (no downforce accounted for)
        acc_lon = (self.force_engine(v) - (v**2 * self.c_drag) ) / self.mass                        
        acc_lon -=  self.c_roll * 9.81                               #rolling resistance
        return min(acc_lon_max, acc_lon)

    def get_min_acc(self,v, acc_lat):
        '''maximum possible deceleration (braking)'''
        n = self.trail_braking / 50
        acc_lon = self.dec_limit * (1 - (np.abs(acc_lat) / self.acc_grip_max)**n)**(1/n)
        acc_lon +=  v**2 * self.c_drag / self.mass
        acc_lon +=  self.c_roll * 9.81 #rolling resistance
        return acc_lon

    def force_engine(self, v):
        P_engine = self.P_engine / 1.3410 * 1000  # from hp to Watt
        return P_engine / v   #tractive force (limited by engine power)

    def get_gear(self, v):
        return v*0


@dataclass
class Track:
    name: str
    border_left: np.ndarray
    border_right: np.ndarray
    min_clearance: float = 0
    
    def __post_init__(self):
        self.position_clearance = self.min_clearance / self.width

    @property
    def width(self):
        return np.sum((self.border_right[:,:2] - self.border_left[:,:2])**2, 1) ** 0.5
    @property
    def slope(self):
        return (self.border_right[:,2] - self.border_left[:,2]) / self.width
    @property
    def left_x(self):
        return self.border_left[:,0]
    @property
    def left_y(self):
        return self.border_left[:,1]
    @property
    def right_x(self):
        return self.border_right[:,0]
    @property
    def right_y(self):
        return self.border_right[:,1]
            
    def get_line_coordinates(self, position: np.ndarray = None) -> np.ndarray:
        return self.border_left + (self.border_right - self.border_left) * np.expand_dims(position, axis=1)

    def get_track_borders(self):
        # return pd.DataFrame([[self.left_x, self.left_y, self.right_x, self.right_y]], columns=['left_x','left_y','right_x','right_y'])

        return pd.DataFrame(
            data = np.column_stack([self.left_x, self.left_y, self.right_x, self.right_y]),
            columns=['left_x','left_y','right_x','right_y'],
            )

    def calc_line(self, position):
        position = np.clip(position, a_min=self.position_clearance, a_max=1-self.position_clearance)
        line = self.get_line_coordinates(position)
        ds = mag(np.diff(line.T,1 ,prepend=np.c_[line[-1]]).T)     #distance from previous
        s = ds.cumsum() - ds[0]
        return line, ds, s

    def get_curvature(self, position):
        line, _, _ = self.calc_line(position)
        dX = np.gradient(line, axis=0)
        ddX = np.gradient(dX, axis=0)
        return mag(np.cross(dX, ddX))/mag(dX)**3 

    def new_line(self, position):
        start = np.random.randint(0, len(position))
        length = np.random.randint(1, 60)
        deviation = np.random.randn() / 10
       
        line_adjust = 1 - np.cos(np.linspace(0, 2*np.pi, length))
        new_line = self.width * 0
        new_line[:length] = line_adjust * deviation
        new_line = np.roll(new_line, start)
        position = position + new_line / self.width

        return np.clip(position, self.position_clearance, 1 - self.position_clearance)

    def race(self, car: Car, position: np.ndarray = None, verbose: bool = False):

        line, ds, distance = self.calc_line(position)

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
        Bt[:,2] = self.slope         #align Bt with the track and normalize
        Bt = Bt / mag(Bt)[:,None]
        Nt = np.cross(Bt, Tt)

        car_axis = lambda x: np.c_[dot(x, Tt), dot(x, Bt), dot(x, Nt)]
        k_car = car_axis(Nk)          #curvature projected in car frame [lon, lat, z]
        g_car = car_axis(np.array([0, 0, 9.81])[None,:])   #direction of gravity in car frame [lon, lat, z]

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
                v1 =  (v_a[j]**2 + 2*acc_lon * ds[i])**0.5
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


        return pd.DataFrame(
            data = np.column_stack((position, distance, line[:,:2], speed, time, a_lat, a_lon )),
            columns=(('race_line_position', 'distance', 'line_x', 'line_y', 'speed', 'time', 'a_lat', 'a_lon' ))
            ) if verbose else time[-1]