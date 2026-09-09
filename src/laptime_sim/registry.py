import re
from pathlib import Path

import geopandas as gpd
import numpy as np

from laptime_sim.car import Car
from laptime_sim.raceline import Raceline
from laptime_sim.track import Track

_ROOT = Path(__file__).resolve().parents[2]
PATH_CARS = _ROOT / "resources" / "cars"
PATH_TRACKS = _ROOT / "resources" / "tracks"
PATH_RESULTS = _ROOT / "resources" / "simulated"


def cars(path_cars: Path | None = None) -> list[Car]:
    """Returns a list of Car instances from a directory of TOML files."""
    path_cars = PATH_CARS if path_cars is None else path_cars
    return [Car.from_toml(file) for file in sorted(path_cars.glob("*.toml"))]


def tracks(path_tracks: Path | None = None) -> list[Track]:
    """Returns a list of Track instances from a directory of parquet files."""
    path_tracks = PATH_TRACKS if path_tracks is None else path_tracks
    if not path_tracks.is_dir():
        raise FileNotFoundError(f"Path {path_tracks} is not a directory.")
    return [Track(layout=gpd.read_parquet(file)) for file in sorted(path_tracks.glob("*.parquet"))]


def _safe_file_name(name: str) -> str:
    """Strips path separators and Windows-invalid characters from a file stem."""
    cleaned = re.sub(r'[<>:"/\\|?*]', "_", name).strip(" ._")
    if not cleaned:
        raise ValueError(f"name {name!r} contains only invalid characters")
    return cleaned


def save_track(track: gpd.GeoDataFrame, name: str) -> Path:
    """
    Writes a track layout to the tracks directory.

    Parameters:
    - track: GeoDataFrame - The track layout with 'track_name' and 'geom_type' columns.
    - name: str - The track name to use for the parquet file.

    Returns:
    - Path: The path the track was written to.
    """
    if "track_name" not in track.columns or "geom_type" not in track.columns:
        raise ValueError("track must have 'track_name' and 'geom_type' columns")
    PATH_TRACKS.mkdir(parents=True, exist_ok=True)
    path = PATH_TRACKS / f"{_safe_file_name(name)}.parquet"
    track.to_parquet(path)
    return path


def results_path(car: Car, track: Track) -> Path:
    """Returns the results file path for a car and track combination."""
    return PATH_RESULTS / f"{car.file_name}_{track.name}_simulated.parquet"


def save_result(raceline: Raceline, car: Car) -> Path:
    """
    Writes the raceline to the results file for the raceline's track and the car.

    Parameters:
    - raceline: Raceline - The raceline to save.
    - car: Car - The car the raceline was simulated for.

    Returns:
    - Path: The path the results were written to.
    """
    path = results_path(car, raceline.track)
    PATH_RESULTS.mkdir(exist_ok=True)
    raceline.dataframe(track_name=raceline.track.name, car_name=car.name).to_parquet(path)
    return path


def load_result(track: Track, car: Car) -> tuple[Raceline, bool]:
    """
    Loads the simulated raceline for a car and track combination.

    Parameters:
    - track: Track - The track to load the raceline for.
    - car: Car - The car to load the raceline for.

    Returns:
    - tuple: (raceline, found) where raceline is a fresh Raceline when no result
      exists, and found indicates whether a stored result was loaded.
    """
    raceline = Raceline(track)
    path = results_path(car, track)
    if not path.exists():
        return raceline, False

    data = gpd.read_parquet(path)
    assert track.name == data.iloc[0].track_name
    data = data.to_crs(data.estimate_utm_crs())
    coordinates = data.get_coordinates(include_z=False).to_numpy(na_value=0)

    assert len(coordinates) == track.len, f"Track length {track.len} != coordinates length {len(coordinates)}"

    if track.is_circular:
        assert np.array_equal(coordinates[0], coordinates[-1]), "First and last coordinates are not the same"

    raceline.set_coordinates(coordinates)
    return raceline, True


def read_results(track: Track, car_names: list[str]) -> gpd.GeoDataFrame:
    """
    Reads all stored racelines for a track, ignoring racelines orphaned by car
    renames (see AGENTS.md).

    Parameters:
    - track: Track - The track to read results for.
    - car_names: list[str] - Names of the cars to include.

    Returns:
    - GeoDataFrame: The stored racelines, empty when no results exist.
    """
    try:
        all_racelines = gpd.read_parquet(PATH_RESULTS, filters=[("track_name", "==", track.name)])
    except ValueError:
        return gpd.GeoDataFrame()
    return all_racelines[all_racelines["car"].isin(car_names)]
