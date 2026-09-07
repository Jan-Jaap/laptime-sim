import time

import laptime_sim
from laptime_sim.simulate import simulate


def test_benchmark(race_car, track):
    """Informational benchmark: run with `uv run pytest tests/test_benchmark.py -s` to see timings."""
    raceline = laptime_sim.Raceline(track)
    simulate(race_car, raceline.get_coordinates(), track.slope)  # JIT warmup

    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        simulate(race_car, raceline.get_coordinates(), track.slope)
    t_sim = (time.perf_counter() - t0) / n

    t0 = time.perf_counter()
    for _ in range(n):
        raceline.try_random_line(race_car)
    t_rand = (time.perf_counter() - t0) / n

    print(
        f"\n{track.name} ({track.len} pts): simulate {t_sim * 1000:.3f} ms/call, "
        f"try_random_line {t_rand * 1000:.3f} ms/iter"
    )
