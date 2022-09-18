# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas
import numpy as np 
#from psopy import _minimize_pso, init_feasible
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import norm
from scipy.interpolate import CubicSpline, griddata
import matplotlib.pyplot as plt
from matplotlib.path import Path as mpath
from matplotlib.patches import PathPatch as mpatch


class Point:
	def __init__(self,x,y):
		self.x = x
		self.y = y



def orientation(p, q, r):
    ## See https://www.geeksforgeeks.org/orientation-3-ordered-points/ 
    ## for details of below formula. 
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
  
    if val == 0: return 0  ## colinear :
    elif val > 0: return 1 #clock wise
    else: return 2          # counterclock wise 


#def intersect(A,B,C,D):
#	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def intersect(p1,q1,p2,q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    ##General case
    if (o1 != o2 and o3 != o4):
        return True; 
    
    onSegment = lambda p, q, r: q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y)
        ## Special Cases 
    ## p1, q1 and p2 are colinear and p2 lies on segment p1q1 
    if o1 == 0 and onSegment(p1, p2, q1): return True
  
    ## p1, q1 and q2 are colinear and q2 lies on segment p1q1 
    if o2 == 0 and onSegment(p1, q2, q1): return True
  
    ## p2, q2 and p1 are colinear and p1 lies on segment p2q2 
    if o3 == 0 and onSegment(p2, p1, q2): return True
  
    ## p2, q2 and q1 are colinear and q1 lies on segment p2q2 
    if o4 == 0 and onSegment(p2, q1, q2): return True
  
    return False ## Doesn't fall in any of the above cases 


    # read starting positions from file
df = pandas.read_csv('WP_POS.csv', index_col='Row Labels')
    


try:
    position = df['Path_position'].values + df['Race_position'].values
except:
    position = df['Path_position'].values + 0

p_Left = mpath(np.c_[df.Left_x, df.Left_y])
p_Right = mpath(np.c_[df.Right_x, df.Right_y])

                    
df.Left_z = savgol_filter(df.Left_z, 121, 5, mode='wrap') # window size 51, polynomial order 2
df.Right_z = savgol_filter(df.Right_z, 121, 5, mode='wrap') # window size 51, polynomial order 2

points = np.c_[np.r_[df.Left_x, df.Right_x], np.r_[df.Left_y, df.Right_y]]
#values = np.r_[df.Left_z, df.Right_z]



Raceline_x = (df.Left_x + (df.Right_x - df.Left_x) * position).values
Raceline_y = (df.Left_y + (df.Right_y - df.Left_y) * position).values


Raceline_x = savgol_filter(Raceline_x, 121, 5, mode='wrap') # window size 51, polynomial order 2
Raceline_y = savgol_filter(Raceline_y, 121, 5, mode='wrap') # window size 51, polynomial order 2

for i, [x0,y0,x1,y1] in enumerate(np.c_[df.Left_x, df.Left_y,df.Right_x, df.Right_y]):
    p1=Point(x0,y0)
    q1=Point(x1,y1)
    for j in range(len(position)):
        p2 = Point(Raceline_x[j-1], Raceline_y[j-1])
        q2 = Point(Raceline_x[j], Raceline_y[j])
        if intersect(p1,q1,p2,q2):
            #find intersection point  (ax+b)
            a1 = (q1.y-p1.y)/(q1.x-p1.x)  #SAC
            a2 = (q2.y-p2.y)/(q2.x-p2.x)  #SBD
            b1 = q1.y - a1 * q1.x         #IAC
            b2 = q2.y - a2 * q2.x         #IBD
            ix = (b2 - b1) / (a1 - a2)
            iy = a1 * ix + b1
            
            position[i] = (ix - p1.x) / (q1.x - p1.x)
            break
        
Raceline_z = (df.Left_z + (df.Right_z - df.Left_z) * position).values




df['Raceline_x'] = Raceline_x
df['Raceline_y'] = Raceline_y
df['Raceline_z'] = Raceline_z
#Raceline_z1 = griddata(points, values, (df.Raceline_x, df.Raceline_y), method='linear')



min_xy = np.min(points,0)
max_xy = np.max(points,0)

#grid_x, grid_y = np.mgrid[min_xy[0]:max_xy[0]:1000j, min_xy[1]:max_xy[1]:1000j]
#grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
#grid_dz = np.gradient(grid_z)




df['Race_position'] = position - df['Path_position'].values

#%% save df            
df.to_csv('WP_POS_clean.csv')
    
#%% plot results    

#plt.figure()
fig, axs = plt.subplots(2, 2)
a1= axs[0,0]
a1.axis('equal')
a1.add_patch(mpatch(p_Left , fill=False, color='r'))
a1.add_patch(mpatch(p_Right, fill=False, color='g'))

#for [x0,y0,x1,y1] in np.c_[df.Left_x, df.Left_y,df.Right_x, df.Right_y]:
#    a1.plot([x0,x1], [y0,y1])
#    
    

#axs[0,0].plot
a1.plot(df.Raceline_x, df.Raceline_y)

axs[0,1].plot(df.Raceline_z, 'x')
axs[0,1].plot(df.Left_z, 'r')
axs[0,1].plot(df.Right_z, 'g')

#axs[1,0].imshow(grid_dz[0].T, extent=(min_xy[0],max_xy[0],min_xy[1],max_xy[1]), origin='lower')
axs[1,0].plot(points[:,0], points[:,1], 'k.', ms=1)


