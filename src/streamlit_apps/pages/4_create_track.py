"""This module creates a streamlit app"""

from typing import Any, Iterable, Optional
import streamlit as st
import folium
import xyzservices.providers as xyz
from streamlit_folium import folium_static, st_folium

# from folium.plugins import Draw

import geopandas as gpd
import shapely
import math
from pathlib import Path

POINT_COUNT = 500
TRANSECT_LENGTH = 12

FILENAME = Path("./temp/silverstone.shp")
PATH_TRACKS = "./tracks/"

start_finish = gpd.points_from_xy([-1.022276814267109], [52.069221331141215], crs="EPSG:4326")


def main() -> None:
    st.set_page_config("Create Track layout")
    st.header("Create Track layout")

    uploaded_file = st.file_uploader("upload track shapefile (zip)", accept_multiple_files=False)

    if uploaded_file is None:
        return

    track: gpd.GeoDataFrame = gpd.read_file(uploaded_file)
    track = track.to_crs(track.estimate_utm_crs())  # utm crs has 1m units required for calculations

    point_count = st.number_input("POINT_COUNT", min_value=100, max_value=5000, value=POINT_COUNT, step=10)
    transect_length = st.number_input("TRANSECT_LENGTH", min_value=1, max_value=100, value=TRANSECT_LENGTH, step=1)
    track_name = st.text_input("Track name", value=Path(uploaded_file.name).stem)
    start_finish = st.text_input("Start finish", value="52.069221331141215,-1.022276814267109")
    start_finish = gpd.array.from_shapely([shapely.Point([float(x) for x in start_finish.split(",")])], crs="EPSG:4326")
    start_finish.to_crs(track.crs)

    inner, outer = track.geometry
    if inner.contains(outer):  # == inside out
        inner, outer = outer, inner
        track = track.reindex(index=[1, 0])

    if track.exterior is not None:  # convert polygons to linearring
        track = track.exterior

    my_map = track.explore(name="track", style_kwds={"color": "blue"})

    # my_map = track.extract_unique_points().explore(m=my_map, name="track_points")

    center_line = create_centerline(track, point_count, start_finish=start_finish)

    gpd.GeoSeries(center_line, crs=track.crs).explore(m=my_map, name="center_line", style_kwds={"color": "blue"})

    transect_lines = transect(center_line, transect_length)
    my_map = gpd.GeoSeries(transect_lines, crs=track.crs).explore(m=my_map, name="transect_lines", style_kwds={"color": "blue"})

    inner_points, outer_points = [], []
    for line in transect_lines.geoms:
        p1, p2 = track.intersection(line)
        inner_points.append(p1)
        outer_points.append(p2)

    new_inner = shapely.LinearRing([(p.x, p.y) for p in inner_points])
    new_outer = shapely.LinearRing([(p.x, p.y) for p in outer_points])
    right = {"geom_type": "left", "geometry": new_inner, "track_name": track_name}
    left = {"geom_type": "right", "geometry": new_outer, "track_name": track_name}
    new_track = gpd.GeoDataFrame.from_dict(data=[left, right], crs=track.crs).to_crs(epsg=4326)
    my_map = new_track.explore(m=my_map, name="new_track", style_kwds={"color": "red"})

    folium.TileLayer(xyz.Esri.WorldImagery).add_to(my_map)
    folium.TileLayer("openstreetmap").add_to(my_map)
    folium.LayerControl().add_to(my_map)
    # Draw(export=True).add_to(my_map)

    folium_static(my_map)

    if st.button("save track"):
        new_track.to_parquet(Path(PATH_TRACKS, track_name + ".parquet"))


def prev_next_iter(my_list: list[Any]) -> Iterable[tuple[Any, Any, Any]]:
    """
    Generates an iterator that yields the previous, current, and next elements in a list.

    Args:
        my_list (List[Any]): The list to iterate over.

    Returns:
        Iterable[Tuple[Any, Any, Any]]: A tuple containing the previous, current, and next elements in the list.

    Example:
        >>> list(prev_next_iter([1, 2, 3, 4]))
        [(1, 2, 3), (2, 3, 4), (3, 4, 1)]
    """
    len_list = len(my_list)
    for i, _ in enumerate(my_list):
        yield my_list[(i - 1) % len_list], my_list[i], my_list[(i + 1) % len_list]


def create_centerline(track, nr_points=600, start_finish=None) -> shapely.LinearRing:
    inner, outer = hacky_offset_curve(track)

    if start_finish:
        offset_outer = outer.line_locate_point(start_finish)[0]
        offset_inner = inner.line_locate_point(start_finish)[0]

    inner = redistribute_vertices(inner, nr_points, offset_inner)
    outer = redistribute_vertices(outer, nr_points, offset_outer)

    return shapely.LinearRing([p.interpolate(0.5, normalized=True) for p in get_divisions(inner, outer).geoms])


def resize_vector(vector: tuple, new_length: float):
    scale_factor = new_length / math.hypot(*vector)
    return (vector[0] * scale_factor, vector[1] * scale_factor)


def transect_func(p0, p1, p2, length=None):
    """
    Generate a transect line based on three points.

    Parameters:
        p0 (tuple): The coordinates of the first point (x0, y0).
        p1 (tuple): The coordinates of the second point (x1, y1).
        p2 (tuple): The coordinates of the third point (x2, y2).

    Returns:
        tuple: A tuple containing the coordinates of the transect line's starting point (x1, y1)\
              and the slope of the transect line (x1 - y0 + y2, y1 + x0 - x2).
    """
    x0, y0, x1, y1, x2, y2 = p0[0], p0[1], p1[0], p1[1], p2[0], p2[1]
    dy = y2 - y0
    dx = x0 - x2

    if length:
        dx, dy = resize_vector((dx, dy), length)

    return ((x1 - dy, y1 - dx)), (x1 + dy, y1 + dx)


def transect(line: shapely.LinearRing, length: Optional[float] = None) -> shapely.MultiLineString:
    """
    Generate a MultiLineString containing transect lines based on a LinearRing.

    Parameters:
        line (shapely.LinearRing): A shapely LinearRing, representing the border.
        length (Optional[float]): The length of the transect lines to be generated.

    Returns:
        shapely.MultiLineString: A MultiLineString, containing the generated transect lines.
    """
    line_coords = list(line.coords)[:-1]
    lines = [transect_func(p0, p1, p2, length) for p0, p1, p2 in prev_next_iter(line_coords)]
    return shapely.MultiLineString(lines=lines)


def hacky_offset_curve(series: gpd.GeoSeries) -> tuple[shapely.LinearRing, shapely.LinearRing]:
    """
    Generate a tuple of two shapely.LinearRings, the first being the inner offset curve, and the second the outer offset curve.

    Parameters:
        series (GeoSeries): A GeoSeries containing a single LineString, representing the border.

    Returns:
        tuple[LinearRing, LinearRing]: A tuple containing the inner and outer offset curves.
    """
    inner, outer = series.geometry
    offset_distance = inner.distance(outer) / 2  # border offsets will touch in the middle
    inner = inner.offset_curve(offset_distance)
    outer = outer.reverse().offset_curve(offset_distance).reverse()
    return inner, outer


def get_divisions(left: shapely.LineString, right: shapely.LineString):
    return shapely.MultiLineString(lines=[div for div in zip(left.coords, right.coords)])


def redistribute_vertices(geom, num_vert: int, offset: float = 0.0):
    if geom.geom_type not in ["LinearRing", "LineString", "MultiLineString"]:
        raise ValueError("unhandled geometry %s", (geom.geom_type,))

    if geom.geom_type == "MultiLineString":
        parts = [redistribute_vertices(part, num_vert) for part in geom]
        return type(geom)([p for p in parts if not p.is_empty])

    # num_vert = int(round(geom.length / distance)) or 1
    offset /= geom.length
    coordinates = [geom.interpolate((float(n) / num_vert + offset) % 1, normalized=True) for n in range(num_vert + 1)]

    if geom.geom_type == "LineString":
        return shapely.LineString(coordinates)

    if geom.geom_type == "LinearRing":
        return shapely.LinearRing(coordinates)


if __name__ == "__main__":
    main()
