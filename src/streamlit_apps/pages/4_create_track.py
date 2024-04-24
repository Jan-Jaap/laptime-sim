"""This module creates a streamlit app"""

# import numpy as np
import streamlit as st
import folium
import xyzservices.providers as xyz
from streamlit_folium import st_folium, folium_static

# from folium.plugins import Draw

import geopandas as gpd
import shapely


def main() -> None:
    st.set_page_config("Create Track layout")
    st.header("Create Track layout")

    # s = st.selectbox(label='Zoom to known track', options=KNOWN_TRACKS)
    # st.file_uploader('upload track shapefile', accept_multiple_files=False)

    track: gpd.GeoDataFrame = gpd.read_file("./temp/silverstone.shp")
    track = track.to_crs(track.estimate_utm_crs())  # utm crs has 1m units required for calculations

    inner, outer = track.geometry

    if outer.contains(inner):  # == inside out
        track = track.reindex([1, 0])

    if track.exterior is not None:  # convert polygons to linearring
        track = track.exterior

    start_finish = gpd.points_from_xy([-1.022276814267109], [52.069221331141215], crs="EPSG:4326").to_crs(track.crs)

    my_map = track.explore(name="track", style_kwds={"color": "blue"})
    my_map = track.extract_unique_points().explore(m=my_map, name="track_points")

    track_offset = hackey_offset_curve(track)
    my_map = track_offset.explore(m=my_map, name="track_offset", style_kwds={"color": "red"})

    outer, inner = track_offset.geometry
    s_inner = inner.line_locate_point(start_finish)
    s_outer = outer.line_locate_point(start_finish)

    inner = redistribute_vertices(inner, 600, offset=s_inner[0])
    outer = redistribute_vertices(outer, 600, offset=s_outer[0])

    midpoints = shapely.LinearRing([p.interpolate(0.5, normalized=True) for p in get_divisions(outer, inner).geoms])
    divisions = gpd.GeoSeries(midpoints, crs=track.crs)

    my_map = divisions.explore(m=my_map, name="divisions", style_kwds={"color": "black"})
    my_map = divisions.extract_unique_points().explore(m=my_map, name="divisions", style_kwds={"color": "black"})

    folium.TileLayer(xyz.Esri.WorldImagery).add_to(my_map)
    folium.TileLayer("openstreetmap").add_to(my_map)
    folium.LayerControl().add_to(my_map)
    # Draw(export=True).add_to(my_map)

    # x = st_folium(my_map, use_container_width=True)
    folium_static(my_map)


def hackey_offset_curve(series):
    outer, inner = series.geometry
    offset_distance = inner.distance(outer) / 2  # border offsets will touch in the middle
    inner = inner.offset_curve(offset_distance)
    # outer = outer.offset_curve(-offset_distance)  # doesn't work as it should, hack below
    outer = outer.reverse().offset_curve(offset_distance).reverse()
    return gpd.GeoSeries([shapely.LinearRing(inner), shapely.LinearRing(outer)], crs=series.crs)


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
