import numpy as np

class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct={}):
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


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
