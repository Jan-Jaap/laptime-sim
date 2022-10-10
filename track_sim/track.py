from dataclasses import dataclass
from typing import Any
import numpy as np

import pandas as pd

import geopandas as gpd
from shapely.geometry import LineString, LinearRing

import plotly.graph_objects as go



@dataclass
class Track:
    name: str
    border_left: np.ndarray
    border_right: np.ndarray
    best_known_raceline: np.ndarray = None
    track_record: float = None
    min_clearance: float = 0
    crs: int = None
    
    def __post_init__(self):
        self.position_clearance = self.min_clearance / self.width
        if self.best_known_raceline is None:
            self.best_known_raceline = self.position_clearance  #hugging left edge
        # self.track_record = self.race()

    @classmethod
    def from_csv(cls, file_name, crs=None):
        df = pd.read_csv(file_name)

        return cls(
            name=file_name, 
            border_left         = df.filter(regex="outer_").values, 
            border_right        = df.filter(regex="inner_").values, 
            best_known_raceline = df.filter(regex="line_").values,
            track_record        = df.Timestamp.iloc[-1] if 'Timestamp' in df.columns else None,
            crs                 = crs
            )

    @classmethod
    def from_geojson(cls, file_name):
        return gpd.read_file(file_name)
        # return cls(
        #     name=file_name, 
        #     border_left         = df.filter(regex="outer_").values, 
        #     border_right        = df.filter(regex="inner_").values, 
        #     best_known_raceline = df.filter(regex="line_").values,
        #     track_record        = df.Timestamp.iloc[-1] if 'Timestamp' in df.columns else None
        #     )

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
    @property
    def line_x(self):
        return self.best_known_raceline[:,0]
    @property
    def line_y(self):
        return self.best_known_raceline[:,1]
            
    def get_line_coordinates(self, position: np.ndarray = None) -> np.ndarray:
        return self.border_left + (self.border_right - self.border_left) * np.expand_dims(position, axis=1)

    # def figure(self):
    #     MODE = 'lines'
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=self.left_x , y=self.left_y ,mode=MODE, name='border_left' ))
    #     fig.add_trace(go.Scatter(x=self.right_x, y=self.right_y,mode=MODE, name='border_right' ))
        
    #     if self.best_known_raceline.size != 0:
    #         fig.add_trace(go.Scatter(x=self.line_x, y=self.line_y,
    #             mode=MODE, name='line', line=dict(width=2, dash='dash')))
        
    #     fig.update_xaxes(showticklabels=False, zeroline=False)
    #     fig.update_yaxes(showticklabels=False, zeroline=False, scaleanchor = "x", scaleratio = 1)
    #     return fig

    def to_geojson(self):
                
        track_dict = dict(
        inner = LinearRing(self.border_left[:,:2].tolist()),
        outer = LinearRing(self.border_right[:,:2].tolist())
        )

        if self.best_known_raceline.size != 0:
            # print('adding raceline')
            track_dict['line'] = LinearRing(self.best_known_raceline[:,:2].tolist())
        
        gdf = gpd.GeoDataFrame(
            dict(
                name=list(track_dict.keys()),
                geometry=list(track_dict.values())
                ), 
            crs=self.crs)

        return gdf.to_crs(gdf.estimate_utm_crs())


    def save_geosjon(self, filename):
        if not filename.endswith('.geosjon'):
            filename += '.geojson'
        self.to_file(filename,  driver='GeoJSON')
