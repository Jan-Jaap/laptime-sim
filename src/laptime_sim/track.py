import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self, Union

import numpy as np
import shapely
from geopandas import GeoDataFrame, GeoSeries, read_parquet
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class Track:
    layout: GeoDataFrame

    def __post_init__(self):
        """ensure layout is always in local utm coordinate system. This give most accurate results"""
        utm_crs = self.layout.estimate_utm_crs()
        super.__setattr__(self, "layout", self.layout.to_crs(utm_crs))

    @classmethod
    def from_parquet(cls, filename: str) -> Self:
        return cls(layout=read_parquet(filename))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.name == other.name

    def normalize_distance(self, distance):
        return distance / self.width

    @classmethod
    def from_name(cls, track_name: str, path: Union[str, os.PathLike]) -> Self:
        return cls(layout=read_parquet(path, filters=[("track_name", "==", track_name)]))

    @functools.cached_property
    def width(self) -> np.ndarray:
        return np.sum((self.left_coords_2d - self.right_coords_2d) ** 2, 1) ** 0.5

    @functools.cached_property
    def slope(self) -> np.ndarray:
        return (self.right_coords[:, 2] - self.left_coords[:, 2]) / self.width

    @functools.cached_property
    def left(self) -> GeoSeries:
        return self.layout[self.layout["geom_type"] == "left"]

    @functools.cached_property
    def right(self) -> GeoSeries:
        return self.layout[self.layout["geom_type"] == "right"]

    @functools.cached_property
    def is_circular(self) -> bool:
        """
        Whether the track is a circular track or not.

        A track is circular if the first and last points are the same [start/finish].
        """
        return self.layout.is_ring.all()

    @functools.cached_property
    def left_coords(self) -> np.ndarray:
        return self.left.get_coordinates(include_z=True).to_numpy(na_value=0)

    @functools.cached_property
    def right_coords(self) -> np.ndarray:
        return self.right.get_coordinates(include_z=True).to_numpy(na_value=0)

    @functools.cached_property
    def left_coords_2d(self) -> np.ndarray:
        return self.left.get_coordinates().to_numpy(na_value=0)

    @functools.cached_property
    def right_coords_2d(self) -> np.ndarray:
        return self.right.get_coordinates().to_numpy(na_value=0)

    @functools.cached_property
    def name(self) -> str:
        return self.layout.track_name[0]

    @functools.cached_property
    def len(self) -> int:
        return len(self.width)

    @functools.cached_property
    def crs(self):
        return self.layout.crs

    @functools.cached_property
    def divisions(self):
        border_left = self.left_coords_2d
        border_right = self.right_coords_2d
        lines = []
        for point_left, point_right in zip(border_left, border_right):
            lines.append(([(point_left), (point_right)]))
        return GeoSeries(shapely.MultiLineString(lines=lines), index=["divisions"], crs=self.crs)

    @functools.cached_property
    def start_finish(self):
        p1, p2 = self.left_coords[0], self.right_coords[0]
        return GeoSeries(shapely.LineString([p1, p2]), crs=self.crs)

    def line_coordinates(self, line_pos: np.ndarray = None) -> np.ndarray:
        return self.left_coords + (self.right_coords - self.left_coords) * np.expand_dims(line_pos, axis=1)

    def initial_line(self, smoothing_window: int = 20, poly_order: int = 5):
        """
        Initializes the raceline by generating a smoothed line of coordinates
        along the track.

        Parameters:
        - track: Track - The track to initialize the raceline on.
        """
        x, y, _ = self.line_coordinates(np.full_like(self.width, 0.5)).T
        smoothed_x = savgol_filter(x, smoothing_window, poly_order, mode="wrap")
        smoothed_y = savgol_filter(y, smoothing_window, poly_order, mode="wrap")
        return np.array([smoothed_x, smoothed_y]).T


def track_list(path_tracks: Path | str):
    path_tracks = Path(path_tracks)
    return [Track.from_parquet(file) for file in path_tracks.glob("*.parquet")]
