import functools

# from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import shapely
from geopandas import GeoDataFrame, GeoSeries, read_parquet
from scipy.signal import savgol_filter


class Track:
    layout: GeoDataFrame

    def __init__(self, layout: GeoDataFrame):
        self.layout = layout.to_crs(layout.estimate_utm_crs())

    @classmethod
    def from_parquet(cls, filename: str) -> Self:
        return cls(layout=read_parquet(filename))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return self.name == other.name

    @functools.cached_property
    def width(self) -> np.ndarray:
        return np.sum((self.left_coords_2d - self.right_coords_2d) ** 2, 1) ** 0.5

    @functools.cached_property
    def slope(self) -> np.ndarray:
        return (self.right_coords[:, 2] - self.left_coords[:, 2]) / self.width

    @functools.cached_property
    def left(self) -> GeoSeries:
        return self.layout[self.layout["geom_type"] == "left"].geometry

    @functools.cached_property
    def right(self) -> GeoSeries:
        return self.layout[self.layout["geom_type"] == "right"].geometry

    @functools.cached_property
    def is_circular(self) -> np.bool:
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

    def centerline(self):
        return (self.left_coords + self.right_coords) / 2

    def initial_line(self, window_length: int = 40, polyorder: int = 4):
        """
        Initializes the raceline by generating a smoothed line of coordinates
        along the track.

        Parameters:
        - track: Track - The track to initialize the raceline on.
        """

        initial_line = savgol_filter(self.centerline(), window_length, polyorder, axis=0, mode="wrap")

        if self.is_circular:
            initial_line[-1] = initial_line[0]  # ensure circular track hack
        return initial_line

    def position_from_coordinates(self, line_coords: np.ndarray) -> np.ndarray:
        if self.is_circular:
            assert (line_coords[0] == line_coords[-1]).all()
            line_coords = line_coords[:-1]
        return np.array(
            [
                loc_line(pl, pr, loc)
                for pl, pr, loc in zip(
                    self.left_coords_2d,
                    self.right_coords_2d,
                    line_coords,
                )
            ]
        )

    def coordinates_from_position(self, line_pos: np.ndarray = None) -> np.ndarray:
        if self.is_circular:
            assert self.len - len(line_pos) == 1
            line_pos = np.append(line_pos, line_pos[0])
        return self.left_coords + (self.right_coords - self.left_coords) * np.expand_dims(line_pos, axis=1)


def track_list(path_tracks: Path | str):
    path_tracks = Path(path_tracks)
    return [Track.from_parquet(file) for file in sorted(path_tracks.glob("*.parquet"))]


def loc_line(point_left, point_right, point_line):
    division = shapely.LineString([(point_left), (point_right)])
    intersect = shapely.Point(point_line)
    return division.project(intersect, normalized=True)
