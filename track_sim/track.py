from dataclasses import dataclass
import numpy as np


def mag(vector):
    return np.sum(vector**2, 1)**0.5


@dataclass
class Track:
    
    outside: np.ndarray
    inside: np.ndarray
    min_clearance: float = 0.1
    new_line_parameters: np.ndarray = None  

    @property
    def width(self):
        return np.sum((self.inside[:,:2] - self.outside[:,:2])**2, 1) ** 0.5

    @property
    def slope(self):
        return (self.inside[:,2] - self.outside[:,2]) / self.width


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


# class Track:
#     new_line_parameters = []
#     def __init__(self, outside, inside):
#         self.width = np.sum((inside[:,:2] - outside[:,:2])**2, 1) ** 0.5
#         self.slope =  (inside[:,2] - outside[:,2]) / self.width
#         self.outside = outside
#         self.inside = inside
#         self.min_clearance = 0.1
#         return
    
#     def calc_line(self, position=None):

#         if position is None:
#             position = self.position
#         position = np.clip(position, a_min = self.min_clearance, a_max = 1-self.min_clearance)
        
#         self.line = self.outside + (self.inside - self.outside) * position[:,None]  
#         self.ds = mag(np.diff(self.line.T,1 ,prepend=np.c_[self.line[-1]]).T)     #distance from previous
       
#         self.s = self.ds.cumsum() - self.ds[0]
#         return

#     def new_line(self, position):
#         start = np.random.randint(0, len(self.width))
# #        start = np.random.randint(3000, 3400)
#         length = np.random.randint(1, 50)
#         deviation = np.random.randn() / 10

#         self.new_line_parameters = dict(start=start, length=length, deviation=deviation)        
        
#         line_adjust = (1 - np.cos(np.linspace(0, 2*np.pi, length))) * deviation

#         new_line = self.width * 0
#         new_line[:length] += line_adjust
#         new_line = np.roll(new_line, start)
#         new_line /= self.width

#         position = position + new_line
#         return position