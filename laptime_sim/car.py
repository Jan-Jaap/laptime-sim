from dataclasses import dataclass
import numpy as np
import json

@dataclass
class Car:
    mass                : float
    P_engine            : float
    acc_limit           : float
    dec_limit           : float
    acc_grip_max        : float
    c_drag              : float
    c_roll              : float
    name                : str = None
    trail_braking       : float = 70

    @classmethod
    def from_file(cls, filename):
        
        with open(filename, 'r') as fp:
            return cls(**json.load(fp))
    
    def get_max_acc(self,v, acc_lat):
        '''maximum possible acceleration (flooring it)'''
        acc_lon_max = self.acc_limit / self.acc_grip_max * (self.acc_grip_max**2 - acc_lat**2)**0.5   #grip circle (no downforce accounted for)
        acc_lon = (self.force_engine(v) - (v**2 * self.c_drag) ) / self.mass                        
        acc_lon -=  self.c_roll * 9.81                               #rolling resistance
        return min(acc_lon_max, acc_lon)

    def get_min_acc(self,v, acc_lat):
        '''maximum possible deceleration (limit braking)'''
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
