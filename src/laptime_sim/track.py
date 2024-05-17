from typing import Self, Union
from geopandas import GeoDataFrame, GeoSeries, read_parquet
from dataclasses import dataclass
import functools
import numpy as np
import shapely
import os


@dataclass(frozen=True)
class Track:
    layout: GeoDataFrame

    def __post_init__(self):
        """ensure layout is always in local utm"""
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
        return np.sum((self.left_coords() - self.right_coords()) ** 2, 1) ** 0.5

    @functools.cached_property
    def slope(self) -> np.ndarray:
        return (self.right_coords()[:, 2] - self.left_coords()[:, 2]) / self.width

    @functools.cached_property
    def left(self) -> GeoSeries:
        return self.layout[self.layout["geom_type"] == "left"]

    @functools.cached_property
    def right(self) -> GeoSeries:
        return self.layout[self.layout["geom_type"] == "right"]

    def left_coords(self, include_z=True) -> np.ndarray:
        return self.left.get_coordinates(include_z=include_z).to_numpy(na_value=0)

    def right_coords(self, include_z=True) -> np.ndarray:
        return self.right.get_coordinates(include_z=include_z).to_numpy(na_value=0)

    @functools.cached_property
    def name(self) -> str:
        return self.layout.track_name[0]

    @functools.cached_property
    def len(self) -> int:
        return len(self.width)

    @property
    def crs(self):
        return self.layout.crs

    @property
    def divisions(self):
        border_left = self.left_coords(include_z=False)
        border_right = self.right_coords(include_z=False)
        lines = []
        for point_left, point_right in zip(border_left, border_right):
            lines.append(([(point_left), (point_right)]))
        return GeoSeries(shapely.MultiLineString(lines=lines), index=["divisions"], crs=self.crs)

    def start_finish(self):
        p1, p2 = self.left_coords()[0], self.right_coords()[0]
        return GeoSeries(shapely.LineString([p1, p2]), crs=self.crs)

    def line_coords(self, line_pos: np.ndarray = None, include_z=True) -> np.ndarray:
        left = self.left_coords(include_z=include_z)
        right = self.right_coords(include_z=include_z)
        return left + (right - left) * np.expand_dims(line_pos, axis=1)
        # return left + (right - left)[:, np.newaxis] * line_pos

    def parametrize_line_coords(self, line_coords: np.ndarray):
        return np.array(
            [
                loc_line(pl, pr, loc)
                for pl, pr, loc in zip(
                    self.left_coords(include_z=False),
                    self.right_coords(include_z=False),
                    line_coords,
                )
            ]
        )


def loc_line(point_left, point_right, point_line):
    division = shapely.LineString([(point_left), (point_right)])
    intersect = shapely.Point(point_line)
    return division.project(intersect, normalized=True)
