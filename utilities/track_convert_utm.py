# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:07:31 2019

@author: JV
"""

from pyproj import Proj, transform
import pandas

df_track = pandas.read_csv('./tracks/20191004_Circuit_Zandvoort.csv')
inProj = Proj(init='epsg:28992')
outProj = Proj(init='epsg:32631')


df_track.inner_x,df_track.inner_y = transform(inProj,outProj,df_track.inner_x.values,df_track.inner_y.values)
df_track.outer_x,df_track.outer_y = transform(inProj,outProj,df_track.outer_x.values,df_track.outer_y.values)
df_track.initial_x,df_track.initial_y = transform(inProj,outProj,df_track.initial_x.values,df_track.initial_y.values)

df_track.to_csv('./tracks/20191004_Circuit_Zandvoort_v2.csv', index = None, header=True)


#print(x2,y2)