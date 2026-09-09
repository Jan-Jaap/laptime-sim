"""Regenerates the golden parity fixtures for the TypeScript optimizer port.

Run: uv run python tests/js/generate_golden.py
The JSON files are committed; tests/js/optimizer.test.mjs replays them against
the TypeScript port and asserts float64 parity.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tests/ for test_registry helpers

import laptime_sim.registry as registry
from laptime_sim import Raceline
from laptime_sim.optimizer import _apply_outcome, run_batch
from laptime_sim.simulate import simulate

OUT = Path(__file__).resolve().parent / "fixtures"


def car_params(car) -> dict:
    return {
        "lat_limit": car.lat_limit,
        "mass": car.mass,
        "acc_limit": car.acc_limit,
        "dec_limit": car.dec_limit,
        "c_drag": car.c_drag,
        "c_roll": car.c_roll,
        "corner_acc": int(car.corner_acc),
        "trail_braking": int(car.trail_braking),
        "P_engine_in_watt": car.P_engine_in_watt,
    }


def geometry_snapshot(raceline: Raceline) -> dict:
    geometry = raceline._geometry
    return {
        "width": geometry.width.tolist(),
        "pos_min": geometry.pos_min.tolist(),
        "pos_max": geometry.pos_max.tolist(),
        "is_circular": geometry.is_circular,
        "idx_len": geometry.idx_len,
        "n_coords": geometry.n_coords,
        "left_coords": geometry.left_coords.tolist(),
        "right_coords": geometry.right_coords.tolist(),
        "slope": geometry.slope.tolist(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    track = registry.tracks()[0]
    car = registry.cars()[0]
    raceline = Raceline(track)
    raceline.simulate(car)
    params = car_params(car)

    # -- simulate: full physics kernel on the initial (left-border) line
    coords = raceline.get_coordinates()
    result = simulate(car, coords, track.slope)
    simulate_fixture = {
        "kind": "simulate",
        "description": f"simulate({track.name}, {car.name}) on the initial line",
        "line_coordinates": coords.tolist(),
        "slope": track.slope.tolist(),
        "car": params,
        "expected": {"laptime": result.laptime},
    }
    (OUT / "simulate.json").write_text(json.dumps(simulate_fixture, indent=1))

    # -- build_batch: candidate construction, clipping, interpolation, simulation
    geometry = raceline._geometry
    idx_len = geometry.idx_len
    locations = np.array([5, idx_len // 3, 2 * idx_len // 3], dtype=np.int64)
    lengths = np.array([30, 5, 60], dtype=np.int64)
    deviations = np.array([0.4, -0.8, 0.15])
    laptimes, positions, perturbations = run_batch(geometry, raceline._position, locations, lengths, deviations, car)
    build_fixture = {
        "kind": "build_batch",
        "description": f"run_batch({track.name}, {car.name}) from the initial line",
        "geometry": geometry_snapshot(raceline),
        "current_position": raceline._position.tolist(),
        "locations": locations.tolist(),
        "lengths": lengths.tolist(),
        "deviations": deviations.tolist(),
        "car": params,
        "expected": {
            "laptimes": laptimes.tolist(),
            "positions": positions.tolist(),
            "perturbations": perturbations.tolist(),
        },
    }
    (OUT / "build_batch.json").write_text(json.dumps(build_fixture, indent=1))

    # -- build_batch on a non-circular geometry (exercises the truncation branch
    #    that no real circuit track covers). The window must avoid near-collinear
    #    point triples: an exactly-zero cross product makes the curvature term
    #    0/0 -> NaN in BOTH implementations (numba and TS), and float noise at a
    #    knife-edge would make the fixture irreproducible.
    from laptime_sim.optimizer import BatchGeometry

    lo, hi = 150, 350
    trim = hi - lo
    open_snapshot = {
        "width": geometry.width[:trim].tolist(),
        "pos_min": geometry.pos_min[:trim].tolist(),
        "pos_max": geometry.pos_max[:trim].tolist(),
        "is_circular": False,
        "idx_len": trim,
        "n_coords": trim,
        "left_coords": geometry.left_coords[:trim].tolist(),
        "right_coords": geometry.right_coords[:trim].tolist(),
        "slope": geometry.slope[:trim].tolist(),
    }
    open_geometry = BatchGeometry(
        width=np.array(open_snapshot["width"]),
        pos_min=np.array(open_snapshot["pos_min"]),
        pos_max=np.array(open_snapshot["pos_max"]),
        is_circular=False,
        idx_len=trim,
        n_coords=trim,
        left_coords=np.array(open_snapshot["left_coords"]),
        right_coords=np.array(open_snapshot["right_coords"]),
        slope=np.array(open_snapshot["slope"]),
    )
    open_locations = np.array([50, 20, trim - 20], dtype=np.int64)
    open_lengths = np.array([30, 80, 60], dtype=np.int64)  # (trim-20) + 60 > trim → truncated
    open_deviations = np.array([0.5, -0.3, 0.2])
    lt, pos, pert = run_batch(open_geometry, np.zeros(trim), open_locations, open_lengths, open_deviations, car)
    open_fixture = {
        "kind": "build_batch",
        "description": f"run_batch on a non-circular trim of {track.name} / {car.name}",
        "geometry": open_snapshot,
        "current_position": np.zeros(trim).tolist(),
        "locations": open_locations.tolist(),
        "lengths": open_lengths.tolist(),
        "deviations": open_deviations.tolist(),
        "car": params,
        "expected": {
            "laptimes": lt.tolist(),
            "positions": pos.tolist(),
            "perturbations": pert.tolist(),
        },
    }
    (OUT / "build_batch_open.json").write_text(json.dumps(open_fixture, indent=1))

    # -- apply_outcome: acceptance with a NaN candidate + decay paths
    base_heatmap = raceline._heatmap.copy()
    base_position = raceline._position.copy()
    initial_best_time = raceline.best_time
    idx_len = geometry.idx_len
    probe = [0.2, 0.5, 0.8]
    batch_positions = np.array([np.full(idx_len, p) for p in probe])
    batch_perturbations = np.array([np.full(idx_len, 0.25 * (i + 1)) for i in range(3)])
    cases = {
        "accept": [np.nan, raceline.best_time * 1.01, raceline.best_time * 0.99],
        "decay": [raceline.best_time * 1.01, raceline.best_time * 1.02, raceline.best_time * 1.03],
        "all_nan": [np.nan, np.nan, np.nan],
    }
    for name, laptimes in cases.items():
        raceline._position = base_position.copy()
        raceline._heatmap = base_heatmap.copy()
        raceline.progress_rate = 1.0
        raceline.best_time = initial_best_time
        improved = _apply_outcome(raceline, np.array(laptimes), batch_positions, batch_perturbations)
        outcome_fixture = {
            "kind": "apply_outcome",
            "description": f"apply_outcome case {name} on {track.name} / {car.name}",
            "geometry": geometry_snapshot(raceline),
            "initial": {
                "best_time": initial_best_time,
                "progress_rate": 1.0,
                "position": base_position.tolist(),
                "heatmap": base_heatmap.tolist(),
            },
            "laptimes": laptimes,
            "positions": batch_positions.tolist(),
            "perturbations": batch_perturbations.tolist(),
            "expected": {
                "improved": improved,
                "best_time": raceline.best_time,
                "progress_rate": raceline.progress_rate,
                "position": raceline._position.tolist(),
                "heatmap": raceline._heatmap.tolist(),
            },
        }
        (OUT / f"apply_outcome_{name}.json").write_text(json.dumps(outcome_fixture, indent=1))

    print(f"fixtures written to {OUT}")


if __name__ == "__main__":
    main()
