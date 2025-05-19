import functools


from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import shapely
from geopandas import GeoDataFrame, GeoSeries, read_parquet
from scipy.signal import savgol_filter


class Track:
    """
    A class representing a track in a racing game.

    Attributes:
    - layout (GeoDataFrame): The layout of the track.
    """

    layout: GeoDataFrame

    def __init__(self, layout: GeoDataFrame):
        """
        Initializes a Track object.

        Parameters:
        - layout (GeoDataFrame): The layout of the track.
        """
        self.layout = layout.to_crs(layout.estimate_utm_crs())

    def __eq__(self, other):
        """
        Checks if two tracks are equal.

        Parameters:
        - other (Track): The other track.

        Returns:
        - bool: Whether the two tracks are equal.
        """
        if not isinstance(other, type(self)):
            return False
        return self.name == other.name

    @functools.cached_property
    def width(self) -> NDArray:
        """
        Calculates the width of the track at each point.

        Returns:
        - NDArray: The width of the track at each point.
        """
        return np.sum((self.left_coords_2d - self.right_coords_2d) ** 2, 1) ** 0.5

    @functools.cached_property
    def slope(self) -> NDArray:
        """
        Calculates the slope of the track at each point.

        Returns:
        - NDArray: The slope of the track at each point.
        """
        return (self.right_coords[:, 2] - self.left_coords[:, 2]) / self.width

    @functools.cached_property
    def left(self) -> GeoSeries:
        """
        Returns the left border of the track.

        Returns:
        - GeoSeries: The left border of the track.
        """
        return self.layout[self.layout["geom_type"] == "left"].geometry

    @functools.cached_property
    def right(self) -> GeoSeries:
        """
        Returns the right border of the track.

        Returns:
        - GeoSeries: The right border of the track.
        """
        return self.layout[self.layout["geom_type"] == "right"].geometry

    @functools.cached_property
    def is_circular(self) -> np.bool:
        """
        Checks if the track is a circular track or not.

        A track is circular if the first and last points are the same [start/finish].

        Returns:
        - bool: Whether the track is circular or not.
        """
        return self.layout.is_ring.all()

    @functools.cached_property
    def left_coords(self) -> NDArray:
        """
        Returns the coordinates of the left border of the track.

        Returns:
        - NDArray: The coordinates of the left border of the track.
        """
        return self.left.get_coordinates(include_z=True).to_numpy(na_value=0)

    @functools.cached_property
    def right_coords(self) -> NDArray:
        """
        Returns the coordinates of the right border of the track.

        Returns:
        - NDArray: The coordinates of the right border of the track.
        """
        return self.right.get_coordinates(include_z=True).to_numpy(na_value=0)

    @functools.cached_property
    def left_coords_2d(self) -> NDArray:
        """
        Returns the 2D coordinates of the left border of the track.

        Returns:
        - NDArray: The 2D coordinates of the left border of the track.
        """
        return self.left.get_coordinates().to_numpy(na_value=0)

    @functools.cached_property
    def right_coords_2d(self) -> NDArray:
        """
        Returns the 2D coordinates of the right border of the track.

        Returns:
        - NDArray: The 2D coordinates of the right border of the track.
        """
        return self.right.get_coordinates().to_numpy(na_value=0)

    @functools.cached_property
    def name(self) -> str:
        """
        Returns the name of the track.

        Returns:
        - str: The name of the track.
        """
        return self.layout.track_name[0]

    @functools.cached_property
    def len(self) -> int:
        """
        Returns the number of points in the track.

        Returns:
        - int: The number of points in the track.
        """
        return len(self.width)

    @functools.cached_property
    def crs(self):
        """
        Returns the coordinate reference system of the track.

        Returns:
        - crs: The coordinate reference system of the track.
        """
        return self.layout.crs

    @functools.cached_property
    def divisions(self):
        """
        Returns the divisions of the track as a GeoSeries.

        Returns:
        - GeoSeries: The divisions of the track as a GeoSeries.
        """
        border_left = self.left_coords_2d
        border_right = self.right_coords_2d
        lines = []
        for point_left, point_right in zip(border_left, border_right):
            lines.append(([(point_left), (point_right)]))
        return GeoSeries(shapely.MultiLineString(lines=lines), index=["divisions"], crs=self.crs)

    @functools.cached_property
    def start_finish(self):
        """
        Returns the start/finish line of the track as a GeoSeries.

        Returns:
        - GeoSeries: The start/finish line of the track as a GeoSeries.
        """
        p1, p2 = self.left_coords[0], self.right_coords[0]
        return GeoSeries(shapely.LineString([p1, p2]), crs=self.crs)

    def centerline(self):
        """
        Returns the centerline of the track as a 2D array of coordinates.

        Returns:
        - NDArray: The centerline of the track as a 2D array of coordinates.
        """
        return (self.left_coords + self.right_coords) / 2

    def initial_line(self, window_length: int = 40, polyorder: int = 4):
        """
        Initializes the raceline by generating a smoothed line of coordinates
        along the track.

        Parameters:
        - window_length (int): The window length for the Savitzky-Golay filter.
        - polyorder (int): The order of the polynomial used for the Savitzky-Golay filter.

        Returns:
        - NDArray: The smoothed line of coordinates.
        """
        initial_line = savgol_filter(self.centerline(), window_length, polyorder, axis=0, mode="wrap")

        if self.is_circular:
            initial_line[-1] = initial_line[0]  # ensure circular track hack
        return initial_line

    def position_from_coordinates(self, line_coords: NDArray) -> NDArray:
        """
        Calculates the position of the raceline at each point.

        Parameters:
        - line_coords (NDArray): The coordinates of the raceline.

        Returns:
        - NDArray: The position of the raceline at each point.
        """
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

    def coordinates_from_position(self, line_pos: NDArray) -> NDArray:
        """
        Calculates the coordinates of the raceline at each point.

        Parameters:
        - line_pos (NDArray): The position of the raceline at each point.

        Returns:
        - NDArray: The coordinates of the raceline at each point.
        """
        if self.is_circular:
            assert self.len - len(line_pos) == 1
            line_pos = np.append(line_pos, line_pos[0])
        return self.left_coords + (self.right_coords - self.left_coords) * np.expand_dims(line_pos, axis=1)


def track_list(path_tracks: Path | str):
    """
    Returns a list of Track instances from a directory of parquet files.

    Parameters:
    - path_tracks (Path | str): The path to the directory of parquet files.

    Returns:
    - list[Track]: A list of Track instances.
    """
    path_tracks = Path(path_tracks)
    if not path_tracks.is_dir():
        raise FileNotFoundError(f"Path {path_tracks} is not a directory.")
    return [Track(layout=read_parquet(file)) for file in sorted(path_tracks.glob("*.parquet"))]


def loc_line(point_left, point_right, point_line):
    """
    Calculates the position of a point on a line defined by two points.

    Parameters:
        point_left (shapely.Point): The left point of the line.
        point_right (shapely.Point): The right point of the line.
        point_line (shapely.Point): The point to calculate the position of.

    Returns:
        float: The position of the point on the line, normalized between 0 and 1.
    """
    division = shapely.LineString([(point_left), (point_right)])
    intersect = shapely.Point(point_line)
    return division.project(intersect, normalized=True)
