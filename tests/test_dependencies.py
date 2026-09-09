import geopandas as gpd
import shapely


def test_geopandas_explore_dependencies_available():
    """Pages 1 and 3 call GeoDataFrame.explore(), which requires folium,
    matplotlib and mapclassify — optional geopandas deps checked lazily at
    call time, so page imports alone do not catch a missing one."""
    gdf = gpd.GeoDataFrame(geometry=[shapely.Point(0, 0)], crs="EPSG:4326")
    explore_map = gdf.explore()
    assert explore_map is not None
