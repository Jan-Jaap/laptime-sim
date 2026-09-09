from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import shapely

from laptime_sim import Car, Raceline, Track, registry


def synthetic_track(name: str = "Test Track") -> gpd.GeoDataFrame:
    """A closed, circular 120-point track with left/right borders and elevation."""
    t = np.linspace(0.0, 2.0 * np.pi, 120)
    lon, lat = 0.001 + 0.001 * np.cos(t), 0.001 + 0.001 * np.sin(t)
    nx, ny = np.sin(t), -np.cos(t)
    offset = 0.0003
    z = 2.0 * np.sin(t)
    left = shapely.LineString(np.column_stack([lon + nx * offset, lat + ny * offset, z]))
    right = shapely.LineString(np.column_stack([lon - nx * offset, lat - ny * offset, z]))
    return gpd.GeoDataFrame.from_dict(
        data=[
            {"track_name": name, "geom_type": "left", "geometry": left},
            {"track_name": name, "geom_type": "right", "geometry": right},
        ],
        crs="EPSG:4326",
    )


def synthetic_car() -> Car:
    return Car(
        name="Test Car",
        mass=1000.0,
        P_engine=150.0,
        acc_limit=4.0,
        dec_limit=9.0,
        lat_limit=10.0,
        c_drag=0.6,
        c_roll=0.012,
        trail_braking=70,
        corner_acc=90,
    )


def test_default_paths_are_package_anchored():
    repo_root = Path(__file__).resolve().parents[1]
    assert repo_root / "resources" / "cars" == registry.PATH_CARS
    assert repo_root / "resources" / "tracks" == registry.PATH_TRACKS
    assert repo_root / "resources" / "simulated" == registry.PATH_RESULTS


def test_save_track_writes_parquet_with_required_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "PATH_TRACKS", tmp_path)
    path = registry.save_track(synthetic_track(), "Test Track")
    assert path == tmp_path / "Test Track.parquet"
    loaded = gpd.read_parquet(path)
    assert set(loaded["geom_type"]) == {"left", "right"}
    assert set(loaded["track_name"]) == {"Test Track"}


def test_save_track_rejects_missing_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "PATH_TRACKS", tmp_path)
    with pytest.raises(ValueError):
        registry.save_track(synthetic_track().drop(columns=["geom_type"]), "Broken")


def test_save_track_sanitizes_unsafe_names(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "PATH_TRACKS", tmp_path)
    path = registry.save_track(synthetic_track(), "../evil/track")
    assert path.parent == tmp_path
    assert not any(sep in path.name for sep in "/\\")
    assert path.exists()
    with pytest.raises(ValueError):
        registry.save_track(synthetic_track(), "...")


def test_result_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "PATH_RESULTS", tmp_path)
    track = Track(synthetic_track())
    car = synthetic_car()

    raceline = Raceline(track)
    raceline.simulate(car)

    path = registry.save_result(raceline, car)
    assert path.parent == tmp_path

    loaded, found = registry.load_result(track, car)
    assert found
    np.testing.assert_allclose(loaded.get_coordinates(), raceline.get_coordinates(), rtol=1e-6, atol=1e-6)
