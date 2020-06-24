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

gravity = 9.81

mag = lambda v: np.sum(v**2, 1)**0.5
dot = lambda u, v: np.einsum('ij,ij->i',u,v)
#c = lambda v:v[:,None]

#import power_curve as rfs

class Track:
    def __init__(self, outside, inside):
        self.width = np.sum((inside[:,:2] - outside[:,:2])**2, 1) ** 0.5
        self.slope =  (inside[:,2] - outside[:,2]) / self.width
        self.outside = outside
        self.inside = inside
        return
    
    def set_line(self, position):
        self.position = np.clip(position, a_min = 0, a_max = 1)
        self.line = self.outside + (self.inside - self.outside) * self.position[:,None]  #first and last should be same point
        self.ds = mag(np.diff(self.line.T,1 ,prepend=np.c_[self.line[-1]]).T)     #distance from previous
        self.s = self.ds.cumsum()                                               #station 0=start/finish
#        self.w_left = track.width * position
#        self.w_right = track.width - self.w_left


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
        acc_lon = (self.force_engine(v) - (v**2 * self.c_drag) ) / self.mass                        
        acc_lon -=  self.c_roll * gravity                               #rolling resistance

        return min(acc_lon_max, acc_lon)

    def force_engine(self, v):
        return self.P_engine / v   #tractive force (limited by engine power)


#    @profile        
    def get_min_acc(self,v, acc_lat):
        '''maximum possible deceleration (braking)'''
        acc_lon =  self.dec_limit * (1 - (acc_lat / self.acc_grip_max)**2)**0.5 #grip circle (no downforce accounted for)
        acc_lon +=  v**2 * self.c_drag / self.mass
        acc_lon +=  self.c_roll * gravity #rolling resistance
        return acc_lon

    def get_gear(self, v):
        return v*0


#     def add_pdf(position, x):
#         
#         [mean, stdev, mag] = x
#         x = np.arange(len(position))+1
#         x05 = len(position)/2
#         y = norm.pdf(np.roll(x,int(mean - x05)),  x05 , stdev) * stdev * mag
#         position = (position + y).clip(0.05,0.95)
#         return position
# =============================================================================
    
class Results(dict):
#    def __init__(self):
#        return
#    
    def __init__(self, *args, **kwargs):
        super(Results, self).__init__(*args, **kwargs)
        self.__dict__ = self
    
    
    
    #    @profile
def simulate(track, car):

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
    
    proj_car_axis = lambda v: np.c_[dot(v, Tt), dot(v, Bt), dot(v, Nt)]
    k_car = proj_car_axis(Nk)          #curvature projected in car axis [lon, lat, z]
    g_car = proj_car_axis(np.array([0, 0, gravity])[None,:])   #direction of gravity in car axis [lon, lat, z]
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
            v1 =  (v0**2 + 2*acc_lon * track.ds[i])**0.5
            v_a[i] = min( v1 ,  v_max[i])
        else:                   #if corner speed was maximal, all grip is used for lateral acceleration (=cornering)
            acc_lon = 0         #no grip available for longitudinal acceleration
            v_a[i] = min( v0 ,  v_max[i])  #speed remains the same

        
        ## max possible speed braking into corners (backwards lap)
        if v_b[j] < v_max[-i]:
            acc_lat = v_b[j]**2 * k_car[-i,1] + g_car[-i,1] 
            acc_lon = car.get_min_acc(v_b[j], acc_lat)
            acc_lon += g_car[j,0]
            v =  (v_b[j]**2 + 2*acc_lon * track.ds[::-1][i])**0.5
            v_b[i] =   min(v  ,  v_max[::-1][i])
            
        else:
            acc_lon = 0
            v_b[i] =  min( v_b[j] ,  v_max[::-1][i])
            
        
    
    v_b = v_b[::-1] #flip te matrix
    
    out = Results()
    out.speed = np.fmin(v_a, v_b)
    out.dt = 2 *  track.ds / (out.speed + np.roll(out.speed,1) ) 
    out.t = out.dt.cumsum()
    out.a_lat = out.speed**2 * Nk[:,1]
    out.a_lon = np.gradient(out.speed, track.s)*out.speed
    out.s = track.s
    out.laptime = out.t[-1]
    return out

def return_dataframe(df, results):
    df['Distance (m)']=results.s
    df['Speed (m/s)'] = results.speed
    df['Longitudinal acceleration (m/s2)'] = results.a_lon
    df['Lateral acceleration (m/s2)'] = results.a_lat
    return df



if __name__ == '__main__':

    import json

    ##ggv diagram /performance envelope parameters
    
    # Peugeot 205 GTi RFS
    with open('./cars/Peugeot_205RFS.json', 'r') as fp:
        race_car = Car(json.load(fp))
    from power_curve_205 import force_engine, gear
    race_car.force_engine = force_engine                     #overloading default function
    race_car.get_gear = gear
    # Peugeot 205 logged data for comparison graph
    df2 = pandas.read_csv('./logged_data/JJ_205RFS/session_zandvoort_circuit_20130604_1504_v2.csv', skiprows = 10) #205 GTi RFS
#    df2 = pandas.read_csv('./logged_data/JJ_205RFS/session_zandvoort_circuit_20140718_2310_v2.csv', skiprows = 10) #205 GTi RFS
    fastest_lap = 50
        
    
# =============================================================================
#     # BMW Z3M Viperwizard
#     with open('./cars/BMW_Z3M.json', 'r') as fp:
#         race_car = Car(json.load(fp))
#     # BMW logged data for comparison graphs
#     df2 = pandas.read_csv('./logged_data/NO_BMWZ3M/session_zandvoort_circuit_20190930_2045_v2.csv', skiprows = 10) # BMW Z3M
#     fastest_lap = 3
# =============================================================================

    
    # read starting positions from file
    df_track = pandas.read_csv('./tracks/20191004_Circuit_Zandvoort.csv')
    track_zandvoort = Track(np.c_[df_track.outer_x.values, df_track.outer_y.values, df_track.outer_z.values],
                            np.c_[df_track.inner_x.values, df_track.inner_y.values, df_track.inner_z.values])
    track_zandvoort.set_line(df_track.initial_position.values)
    
    results = simulate(track_zandvoort, race_car)
  
    print('Simulated laptime = {:02.0f}:{:05.02f} - {}'.format(results.laptime%3600//60, results.laptime%60, race_car.name))

#%% save results
    df_track = return_dataframe(df_track, results)
    df_track.to_csv(race_car.name+'_Zandvoort_simulated.csv', index = None, header=True)



#%% Plot results using plotly
    #    import matplotlib.pyplot as plt
    from plotly.subplots import make_subplots
#    import plotly_express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = 'iframe'

    #discard all data without lap #
    df2 = df2[df2['Lap #']>0]
    
    
    
    fig = make_subplots(rows=6, cols=1, vertical_spacing=0.05,
                        specs=[[{"secondary_y": True}],[{}],[{}],[{}],[{}],[{}]],
                        subplot_titles=(
                                "Speed & delta T",
                                "Gear",
                                "Track Layout and racing line",
                                "Lateral/longitudinal GG-diagram", 
                                "Longitudinal GV-diagram", 
                                "Lateral gv-diagram",
                                ),
                        )
    fig.update_layout(showlegend=False,
                      height=4000,
                      title_text = race_car.name
                      )
                    
    row = 3; 
    col = 1
    fig.add_scatter(x=track_zandvoort.line[:,0],y=track_zandvoort.line[:,1], mode='lines+markers',row=row,col=col)
    fig.add_scatter(x=track_zandvoort.outside.T[0], y=track_zandvoort.outside.T[1], line_color='black',row=row,col=col)
    fig.add_scatter(x=track_zandvoort.inside.T[0], y=track_zandvoort.inside.T[1], line_color='black',row=row,col=col)
    fig.update_xaxes(showticklabels=False, zeroline=False, row=row, col=col)
    fig.update_yaxes(showticklabels=False, zeroline=False, scaleanchor = "x", scaleratio = 1, row=row, col=col)
                  
    # Lateral/longitudinal GG-diagram
    row += 1
    fig.add_scatter(x=df2[str_lat], y=df2[str_lon], row=row,col=col)
    fig.add_scatter(x=results.a_lat, y=results.a_lon, row=row,col=col)
    fig.update_xaxes(title_text='Lateral acceleration [m/s²]',row=row,col=col)
    fig.update_yaxes(title_text='Longitudinal acceleration [m/s²]',row=row,col=col)

     # Longitudinal gv-diagram
    row += 1
    fig.add_scatter(x=df2[str_lon], y=df2[str_spd]*3.6, row=row,col=col)
    fig.add_scatter(x=results.a_lon, y=results.speed*3.6, row=row,col=col)
    fig.update_xaxes(title_text='Longitudinal acceleration [m/s²]',row=row,col=col, anchor='x')
    fig.update_yaxes(title_text='Speed [km/hr]',row=row,col=col)
#     ax3.set_ylabel('Velocity [m/s]')

    # Lateral gv-diagram
    row += 1
    fig.add_scatter(x=df2[str_lat], y=df2[str_spd]*3.6, row=row,col=col)
    fig.add_scatter(x=results.a_lat, y=results.speed*3.6, row=row,col=col)
    fig.update_xaxes(title_text='Lateral acceleration [m/s²]',row=row,col=col, anchor='x')
    fig.update_yaxes(title_text='Speed [km/hr]',row=row,col=col)

    #discard all data except fastest lap
    df2 = df2[df2['Lap #']==fastest_lap]
    df2[str_dist] -= df2[str_dist].values[0]
    df2[str_dist] *= results.s[-1] / df2[str_dist].iloc[-1]
    df2[str_time] -= df2[str_time].values[0]
         
    
    t1 = results.t
    t2 = df2[str_time].values
    t1 = np.interp(df2[str_dist], results.s, t1)

    #plot speed
    row = 1
    fig.add_scatter(x=df2[str_dist], y=np.gradient(t2-t1), fill='tozeroy', row=row,col=col, secondary_y=True)
    fig.add_scatter(x=results.s, y=results.speed*3.6,row=row,col=col)
    fig.add_scatter(x=df2[str_dist], y=df2[str_spd]*3.6,row=row,col=col)
    fig.update_xaxes(title_text='Distance [m]',row=row,col=col)
    fig.update_yaxes(title_text='Speed [km/hr]', row=row,col=col, secondary_y=False)
    fig.update_yaxes(title_text='Delta T', row=row,col=col, secondary_y=True)

    #Gears
    row = 2
    fig.add_scatter(x=results.s, y=race_car.get_gear(results.speed),row=row,col=col)
#    fig.update_yaxes(row=row, col=col,
##                     tickmode = 'array',
##                     tickvals = [0, 1, 2, 3, 4, 5, 6, 7],
##                     ticktext = ['', 'First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth'],
#                     )

    fig.show()
