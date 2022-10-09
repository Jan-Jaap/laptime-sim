from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.graph_objects as go

@dataclass
class Track:
    name: str
    border_left: np.ndarray
    border_right: np.ndarray
    best_known_raceline: np.ndarray = None
    track_record: float = None
    min_clearance: float = 0
    
    def __post_init__(self):
        self.position_clearance = self.min_clearance / self.width
        if self.best_known_raceline is None:
            self.best_known_raceline = self.position_clearance  #hugging left edge

    @classmethod
    def from_csv(cls, file_name):
        df = pd.read_csv(file_name)

        return cls(
            name=file_name, 
            border_left         = df.filter(regex="outer_").values, 
            border_right        = df.filter(regex="inner_").values, 
            best_known_raceline = df.filter(regex="line_").values,
            track_record        = df.Timestamp.iloc[-1] if 'Timestamp' in df.columns else None
            )
    
    
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

    def figure(self):
        MODE = 'lines'
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.left_x , y=self.left_y ,mode=MODE, name='border_left' ))
        fig.add_trace(go.Scatter(x=self.right_x, y=self.right_y,mode=MODE, name='border_right' ))
        
        if self.best_known_raceline.size != 0:
            fig.add_trace(go.Scatter(x=self.line_x, y=self.line_y,
                mode=MODE, name='line', line=dict(width=2, dash='dash')))
        
        fig.update_xaxes(showticklabels=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, zeroline=False, scaleanchor = "x", scaleratio = 1)
        return fig