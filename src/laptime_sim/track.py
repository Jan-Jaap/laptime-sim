from geopandas import GeoDataFrame, GeoSeries, read_parquet
from dataclasses import dataclass
from laptime_sim.geodataframe_operations import divisions
import functools
import numpy as np


@dataclass
class Track:
    layout: GeoDataFrame

    @classmethod
    def from_parquet(cls, filename: str):
        return cls(layout=read_parquet(filename))

    @functools.cached_property
    def _position_clearance(self):
        return self.min_clearance / self.width

    @functools.cached_property
    def width(self):
        return np.sum((self.left_coords() - self.right_coords())**2, 1) ** 0.5

    @functools.cached_property
    def slope(self):
        return (self.right_coords()[:, 2] - self.left_coords()[:, 2]) / self.width

    @property
    def left(self) -> GeoSeries:
        return self.layout.left

    @property
    def right(self) -> GeoSeries:
        return self.layout.right

    def left_coords(self, include_z=True):
        return self.left.get_coordinates(include_z=include_z).to_numpy(na_value=0)

    def right_coords(self, include_z=True):
        return self.right.get_coordinates(include_z=include_z).to_numpy(na_value=0)

    @property
    def name(self):
        return self.layout.name[0]

    @property
    def len(self):
        return len(self.width)

    @property
    def crs(self):
        return self.layout.crs

    @property
    def divisions(self):
        return GeoSeries(divisions(self.layout), index=['divisions'], crs=self.crs)
