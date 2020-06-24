# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:52:24 2019

@author: JV
"""

import numpy as np
#from scipy.interpolate import interpn
import matplotlib.pyplot as plt
 

dyno = np.array([
[1750,9.019276833],
[2000,101.0483693],
[2250,108.700551],
[2500,112.1641771],
[2750,122.6147406],
[3000,129.8723474],
[3250,132.0012694],
[3500,132.554168],
[3750,138.2275087],
[4000,143.0239162],
[4250,143.0325873],
[4500,143.0327722],
[4750,150.207931],
[5000,156.856852],
[5250,158.2287554],
[5500,160.3401915],
[5750,156.1857945],
[6000,151.8116606],
[6250,143.5577314],
[6500,128.3819112],
[6750,100.6101314],
[7000,74.77153063]]
)

#rpm = dyno.T[0]
#torque = dyno.T[1]


r_gears = np.array([2.923,1.850,1.360,1.069,0.865])
#r_gears = np.array([2.923,1.850,1.41,1.15,0.92])
r_final_drive = 3.688  #not sure if correct
r_wheels = 31.5
r_drivetrain = r_gears[:,None] * r_final_drive * r_wheels
force_factor = 9.6  # 1/m

speed = np.arange(6,70)

#rpm = speed * r_gears[:,None] * r_final_drive * r_wheels
rpm = speed * r_drivetrain
#torque = np.interp(rpm, dyno.T[0],  dyno.T[1], float('NaN'),float('NaN'))
torque = np.interp(rpm, dyno.T[0],  dyno.T[1],0,0)

F_engine = torque * r_drivetrain / force_factor
gear_selected = np.argmax(F_engine,0)+1
F_engine_max = np.nanmax(F_engine, 0)

def force_engine(v):
    return np.interp(v,speed,  F_engine_max)

def gear(v):
    return np.rint(np.interp(v,speed,  gear_selected))


if __name__ == '__main__':
    mass_car = 1045
    c_aero = 0.5 * 0.34 * 1.22 * 1.78
    c_roll = 0.016

    
    #%% plot result
    F_drag =  c_aero * speed ** 2 + c_roll * 9.81 * mass_car
    F_car = force_engine(speed)
    gear = gear(speed)
    acc_car = (F_car - F_drag ) / mass_car
    plt.plot(speed, F_car.T)
    plt.plot(speed, gear.T*1000)
    plt.plot(speed, F_drag.T)
    
    
    #rpm = rpm[rpm<7000]




