# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 14:31:52 2019

@author: JV
"""

import pandas
from datetime import datetime

#"C:\Users\JV\Documents\programming\LapTimeSim_Python\gis_data\Spa\Spa francorchamps.csv"
filename='../gis_data/spa/Spa francorchamps.csv'
#filename='../gis_data/woonwerk/woonwerk.csv'
track_name='spa'

df_gis = pandas.read_csv(filename).fillna('x')
#df_sorted = df_gis.sort_values('fid', ascending=False)
df_out = df_gis.pivot(index='TR_SEGMENT', columns='fid2', values=['X','Y','Z'])
df_out = df_out.reorder_levels([1,0], axis=1)
df_out.columns.set_levels(['inner','outer'],level=0,inplace=True)
df_out.columns.set_levels(['x','y','z'],level=1,inplace=True)
df_out.sort_index(axis=1, inplace=True)
#df_out.sort_index(axis='index', ascending=False, inplace=True)
df_out.columns = ['_'.join(col).strip() for col in df_out.columns.values]
filename_out = '{}_{}.csv'.format(datetime.now(tz=None).strftime('%Y%m%d'), track_name)
df_out.to_csv(filename_out, index=False)
# df = pandas.read_csv(filename_out)
