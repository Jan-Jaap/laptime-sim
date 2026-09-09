import numpy as np
import pytest

from laptime_sim import Raceline
from laptime_sim.optimizer import anneal_step, optimize


def test_anneal_step_is_deterministic_with_seed(race_car, track):
    results = []
    for _ in range(2):
        raceline = Raceline(track)
        raceline.simulate(race_car)
        rng = np.random.default_rng(42)
        for _ in range(10):
            anneal_step(raceline, race_car, 20, rng=rng)
        results.append((raceline.best_time, raceline._position.copy()))
    assert results[0][0] == results[1][0]
    np.testing.assert_array_equal(results[0][1], results[1][1])


def test_try_random_lines_delegates_to_anneal_step(race_car, track, monkeypatch):
    calls = []

    def fake_step(rl, car, batch_size, rng=None):
        calls.append(batch_size)

    monkeypatch.setattr("laptime_sim.raceline.anneal_step", fake_step)
    raceline = Raceline(track)
    raceline.try_random_lines(race_car, 5)
    assert calls == [5]


def test_optimize_yields_status_and_finishes(race_car, track, monkeypatch):
    import laptime_sim.optimizer as optimizer_module

    def fake_step(raceline, car, batch_size, rng=None):
        raceline.progress_rate *= 0.5

    monkeypatch.setattr(optimizer_module, "anneal_step", fake_step)
    saved = []

    raceline = Raceline(track)
    raceline.simulate(race_car)
    statuses = list(
        optimize(
            raceline,
            race_car,
            batch_size=7,
            tolerance=0.1,
            update_every=0.0,
            save_every=0.0,
            on_save=lambda: saved.append(True),
        )
    )

    assert len(statuses) == 4  # progress_rate: 1.0 → 0.5 → 0.25 → 0.125 → 0.0625 < 0.1
    assert all(status["finished"] is False for status in statuses[:-1])
    assert statuses[-1]["finished"] is True
    assert statuses[-1]["candidates"] == 28
    assert len(saved) == 4


def test_optimize_saves_before_consumer_breaks(race_car, track, monkeypatch):
    """Real consumers break out of the loop at the finished yield — the final
    tick's on_save must already have run by then."""
    monkeypatch.setattr(
        "laptime_sim.optimizer.anneal_step",
        lambda raceline, car, batch_size, rng=None: setattr(raceline, "progress_rate", raceline.progress_rate * 0.5),
    )
    saved = []
    raceline = Raceline(track)
    raceline.simulate(race_car)
    for status in optimize(
        raceline,
        race_car,
        batch_size=1,
        update_every=0.0,
        save_every=0.0,
        on_save=lambda: saved.append(True),
    ):
        if status["finished"]:
            break
    assert len(saved) == 4


def test_optimize_rejects_unsimulated_raceline(race_car, track, monkeypatch):
    monkeypatch.setattr(
        "laptime_sim.optimizer.anneal_step",
        lambda raceline, car, batch_size, rng=None: setattr(raceline, "progress_rate", raceline.progress_rate * 0.5),
    )
    with pytest.raises(ValueError, match="simulate"):
        list(optimize(Raceline(track), race_car, batch_size=1, update_every=0.0, save_every=0.0))
