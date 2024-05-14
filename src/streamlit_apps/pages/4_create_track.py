"""This module creates a streamlit app"""

from typing import Any, Iterable, Optional
import streamlit as st
import folium
import xyzservices.providers as xyz
from streamlit_folium import folium_static

# from folium.plugins import Draw
# from icecream import ic

import geopandas as gpd
import shapely
import math
from pathlib import Path

POINT_COUNT = 500
TRANSECT_LENGTH = 12

PATH_TRACKS = "./tracks/"


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

    start_finish = track.to_crs(epsg=4326).geometry.iloc[0].boundary.interpolate(0)
    start_finish_input_text = st.text_input("Start finish", value=f"{start_finish.y:.6f},{start_finish.x:.6f}")
    start_finish = gpd.GeoSeries(shapely.Point([float(x) for x in start_finish_input_text.split(",")][::-1]), crs="EPSG:4326")
    start_finish = start_finish.to_crs(track.crs)

    inner, outer = track.geometry
    if inner.contains(outer):  # == inside out
        inner, outer = outer, inner
        track = track.reindex(index=[1, 0])

    if track.exterior is not None:  # convert polygons to linearring
        track = track.exterior
        inner, outer = track.geometry

    center_line = create_centerline(track, point_count, start_finish=start_finish)

    transect_lines = transect(center_line, transect_length)

    inner_points, outer_points = [], []
    for line in transect_lines.geoms:
        p1, p2 = track.intersection(line)

        if inner.has_z:
            s_inner = drop_z(inner).project(p1)
            p1 = inner.interpolate(s_inner)  # to preserve z coordinates

        if outer.has_z:
            s_outer = drop_z(outer).project(p2)
            p2 = outer.interpolate(s_outer)

        inner_points.append(p1)
        outer_points.append(p2)

    new_inner = shapely.LinearRing(inner_points)
    new_outer = shapely.LinearRing(outer_points)

    left = {"track_name": track_name, "geom_type": "left", "geometry": new_outer}
    right = {"track_name": track_name, "geom_type": "right", "geometry": new_inner}
    new_track = gpd.GeoDataFrame.from_dict(data=[left, right], crs=track.crs).to_crs(epsg=4326)

    my_map = track.explore(name="track", style_kwds={"color": "black"})
    track.extract_unique_points().explore(m=my_map, name="track_points")
    gpd.GeoSeries(center_line, crs=track.crs).explore(m=my_map, name="center_line", style_kwds={"color": "blue"})
    gpd.GeoSeries(transect_lines, crs=track.crs).explore(m=my_map, name="transect_lines", style_kwds={"color": "blue"})
    gpd.GeoSeries(transect_lines.geoms[0], crs=track.crs).explore(
        m=my_map,
        name="start_finish",
        style_kwds=dict(
            color="red",
            weight=5,
        ),
    )

    # new_track.explore(m=my_map, name="new_track", style_kwds={"color": "red"})

    folium.TileLayer(xyz.Esri.WorldImagery).add_to(my_map)
    folium.TileLayer("openstreetmap").add_to(my_map)
    folium.LayerControl().add_to(my_map)
    # Draw(export=True).add_to(my_map)

    folium_static(my_map)

    file_name = Path(PATH_TRACKS, track_name + ".parquet")
    if st.button(f"save track: ({file_name})"):
        new_track.to_parquet(file_name)

    with st.expander("Track layout"):
        st.write(new_track)


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
    # inner, outer = track
    inner, outer = hacky_offset_curve(track)
    if start_finish is not None:
        offset_outer = outer.line_locate_point(start_finish)[0]
        offset_inner = inner.line_locate_point(start_finish)[0]

    inner = redistribute_vertices(inner, nr_points, offset_inner)
    outer = redistribute_vertices(outer, nr_points, offset_outer)

    # return shapely.LinearRing(
    #     [((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2) for (x1, y1, z1), (x2, y2, z2) in zip(inner.coords, outer.coords)]
    # )

    return shapely.LinearRing([((x1 + x2) / 2, (y1 + y2) / 2) for (x1, y1), (x2, y2) in zip(inner.coords, outer.coords)])


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


def drop_z(geom: shapely.LineString) -> shapely.LineString:
    return shapely.geometry.LineString([(x, y) for x, y, _ in geom.coords])


if __name__ == "__main__":
    main()
