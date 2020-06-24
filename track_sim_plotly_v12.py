# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np 
import pandas


#Racechrono csv.v2 Headers
rc_header = dict(
        speed  = 'Speed (m/s)',
        distance = 'Distance (m)',
        time = 'Time (s)',
        a_lat = 'Lateral acceleration (m/s2)',
        a_lon = 'Longitudinal acceleration (m/s2)',
        )

gravity = 9.81

mag = lambda v: np.sum(v**2, 1)**0.5
dot = lambda u, v: np.einsum('ij,ij->i',u,v)

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
        n = self.trail_braking / 50
        acc_lon = self.dec_limit * (1 - (np.abs(acc_lat) / self.acc_grip_max)**n)**(1/n)
        acc_lon +=  v**2 * self.c_drag / self.mass
        acc_lon +=  self.c_roll * gravity #rolling resistance
        return acc_lon

    def get_gear(self, v):
        return v*0

   
class Results(dict):
    def __init__(self, *args, **kwargs):
        super(Results, self).__init__(*args, **kwargs)
        self.__dict__ = self
    

    
#@profile
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
    
    results = Results()
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

if __name__ == '__main__':

    import json

    ##ggv diagram /performance envelope parameters
    
# =============================================================================
#     # Peugeot 205 GTi RFS
#     with open('./cars/Peugeot_205RFS.json', 'r') as fp:
#         race_car = Car(json.load(fp))
#      #overloading default functions with cars specific functions
#     from power_curve_205 import force_engine, gear
#     race_car.force_engine = force_engine                    
#     race_car.get_gear = gear
#     
#     # Peugeot 205 logged data for comparison graph
#     filename_log = './logged_data/JJ_205RFS/session_zandvoort_circuit_20130604_1504_v2.csv'
#     fastest_lap = 50
# =============================================================================
    
    
    # BMW Z3M Viperwizard
    with open('./cars/BMW_Z3M.json', 'r') as fp:
        race_car = Car(json.load(fp))
    # BMW logged data for comparison graphs
    filename_log = './logged_data/NO_BMWZ3M/session_zandvoort_circuit_20190930_2045_v2.csv'
    fastest_lap = 1

    
    # read starting positions from file
    nr_iterations = 0
    optimization_results = []
    try:

        df_track = pandas.read_csv(race_car.name+'_Zandvoort_simulated.csv')
        race_line = df_track['Optimized line'].values
    except:
        df_track = pandas.read_csv('./tracks/20191030_Circuit_Zandvoort.csv')
        race_line = df_track.initial_position.values
    
    
    
    track = Track(np.c_[df_track.outer_x.values, df_track.outer_y.values, df_track.outer_z.values],
                            np.c_[df_track.inner_x.values, df_track.inner_y.values, df_track.inner_z.values])
    
    
    results = simulate(race_car, track, race_line)
    print('{} - Simulated laptime = {:02.0f}:{:05.02f}'.format(race_car.name, results.laptime%3600//60, results.laptime%60))

    optimize_yn = input('Start line optimization? [y/N]')
    optimize_results = []
    try:
        while optimize_yn == 'y' or optimize_yn == 'Y':
            new_race_line = track.new_line(results.race_line)
            results_temp = simulate(race_car, track, new_race_line)
            nr_iterations += 1
            if results_temp.laptime < results.laptime:
                results = results_temp
#                print('Simulated laptime = {:02.0f}:{:06.03f}'.format(results.laptime%3600//60, results.laptime%60))
#                print(results.new_line_parameters)
                optimize_results.append([nr_iterations, results.laptime])
    except KeyboardInterrupt:
        print('Interrupted by CTRL+C, saving progress')
        optimize_results = np.array(optimize_results)






#%% save results
    df_track = return_dataframe(df_track, results)
    df_track.to_csv(race_car.name+'_Zandvoort_simulated.csv', index = None, header=True)


#%% prep data
    
    #discard all data without lap #
    df_logged = pandas.read_csv(filename_log, skiprows = 10)
    df_logged = df_logged[df_logged['Lap #']>0]
    df_logged[rc_header['speed']] *= 3.6
    results.speed *= 3.6  #convert speed from m/s to km/h

    title = '{} - Simulated laptime = {:02.0f}:{:05.02f}'.format(race_car.name, results.laptime%3600//60, results.laptime%60)
    print(title)

#%% Plot results using plotly
    from plotly.subplots import make_subplots

    import utm
    import plotly.io as pio
    pio.renderers.default = 'iframe'

    try:
        import matplotlib.pyplot as plt
        plt.plot(optimize_results.T[0], optimize_results.T[1], '-')
    except:
        pass

    

    
    fig = make_subplots(rows=5, cols=1, vertical_spacing=0.025,
                        specs=[[{"secondary_y": True}],[{}],[{}],[{}],[{}]],
                        subplot_titles=(
                                "Speed & delta T",
                                "Track Layout and racing line",
                                "Lateral/longitudinal GG-diagram", 
                                "Lateral GV-diagram",
                                "Longitudinal GV-diagram", 
                                ),
                        )
    fig.update_layout(showlegend=False,
                      height=4000,
                      title_text = title,
                      title_font_size = 30
                      )

                    
    row = 2; 
    col = 1

    fig.add_scatter(name='Track Outside', x=df_track.outer_x, y=df_track.outer_y, text=results.s, line_color='black', row=row, col=col)
    fig.add_scatter(name='Track Inside', x=df_track.inner_x, y=df_track.inner_y, text=results.s, line_color='black', row=row, col=col)
#    fig.add_scatter(x=df_track.initial_x, y=df_track.initial_y, line_color='black', line_dash='dash', row=row,col=col)
    fig.add_scatter(name='Simulated optimal raceline', x=track.line[:,0],y=track.line[:,1], text=results.speed, hovertemplate='Speed:%{text:.2f}km/h', mode='lines+markers', row=row,col=col)
    fig.update_xaxes(showticklabels=False, zeroline=False, row=row, col=col)
    fig.update_yaxes(showticklabels=False, zeroline=False, scaleanchor = "x2", scaleratio = 1, row=row, col=col)

                  
    # Lateral/longitudinal GG-diagram
    row += 1
    fig.add_scatter(name='Logged accelerations', x=df_logged[rc_header['a_lat']], y=df_logged[rc_header['a_lon']], row=row,col=col)
    fig.add_scatter(name='Simulated accelerations', x=results.a_lat, y=results.a_lon, row=row,col=col)
    fig.update_xaxes(title_text='Lateral acceleration [m/s²]',row=row,col=col)
    fig.update_yaxes(title_text='Longitudinal acceleration [m/s²]',row=row,col=col)

    # Lateral gv-diagram
    row += 1
    fig.add_scatter(name='Logged accelerations', x=df_logged[rc_header['a_lat']], y=df_logged[rc_header['speed']], row=row,col=col)
    fig.add_scatter(name='Simulated accelerations', x=results.a_lat, y=results.speed, row=row,col=col)
    fig.update_xaxes(title_text='Lateral acceleration [m/s²]',matches='x3',row=row,col=col)
    fig.update_yaxes(title_text='Speed [km/hr]',row=row,col=col)

     # Longitudinal gv-diagram
    row += 1
    fig.add_scatter(x=df_logged[rc_header['a_lon']], y=df_logged[rc_header['speed']], row=row,col=col)
    fig.add_scatter(x=results.a_lon, y=results.speed, row=row,col=col)
    fig.update_xaxes(title_text='Longitudinal acceleration [m/s²]',row=row,col=col)
    fig.update_yaxes(title_text='Speed [km/hr]',matches='y5',row=row,col=col)
#     ax3.set_ylabel('Velocity [m/s]')

    #discard all data except fastest lap
    df_logged = df_logged[df_logged['Lap #']==fastest_lap]
    df_logged[rc_header['distance']] -= df_logged[rc_header['distance']].values[0]
    df_logged[rc_header['distance']] *= results.s[-1] / df_logged[rc_header['distance']].iloc[-1]
    df_logged[rc_header['time']] -= df_logged[rc_header['time']].values[0]
         
    
    t1 = results.t
    t2 = df_logged[rc_header['time']].values
    t1 = np.interp(df_logged[rc_header['distance']], results.s, t1)


    #plot speed
    row = 1
    fig.add_scatter(x=df_logged[rc_header['distance']], y=np.gradient(t2-t1), fill='tozeroy', row=row,col=col, secondary_y=True)
    fig.add_scatter(name='Simulated optimal raceline', x=results.s, y=results.speed, hovertemplate='Speed:%{y:.2f}km/h', row=row,col=col)
#    fig.add_trace(go.Scatter(x=results.s, y=results.v_max*3.6),row=row,col=col)
    fig.add_scatter(name='Racechrono GPS log', x=df_logged[rc_header['distance']], y=df_logged[rc_header['speed']], hovertemplate='Speed:%{y:.2f}km/h', row=row,col=col)
    fig.update_xaxes(title_text='Distance [m]',row=row,col=col)
    fig.update_yaxes(title_text='Speed [km/hr]', row=row,col=col, secondary_y=False)
    fig.update_yaxes(title_text='Delta T', range=[-0.03, 0.2], row=row,col=col, secondary_y=True)

    #Gears
    fig.add_scatter(name='Gear', x=results.s, y=results.gear,hovertemplate='Gear:%{y:i}', row=row,col=col)

    row = 2; 

    [x_logged_utm, y_logged_utm] = utm.from_latlon( df_logged['Latitude (deg)'].values, df_logged['Longitude (deg)'].values)[:2]
    fig.add_scatter(name='Racechrono GPS log', 
                    x=x_logged_utm, y=y_logged_utm, 
                    line_color='black', 
                    text=df_logged[rc_header['speed']], 
                    hovertemplate='Speed:%{text:.2f}km/h', 
                    line_dash='dot', 
                    mode='lines+markers',  
                    row=row,col=col)
    fig.show()
