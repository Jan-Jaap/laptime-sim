import numpy as np
import pytest

from laptime_sim import Raceline, Track
from laptime_sim.optimizer import accept_result, anneal_step, optimize
from laptime_sim.raceline import BORDER_CLEARANCE_M


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


def _synthetic_env(monkeypatch, tmp_path):
    from test_registry import synthetic_car, synthetic_track

    from laptime_sim import registry

    monkeypatch.setattr(registry, "PATH_RESULTS", tmp_path)
    track = Track(synthetic_track())
    return track, synthetic_car()


def test_accept_result_saves_when_nothing_stored(monkeypatch, tmp_path):
    from laptime_sim import registry

    track, car = _synthetic_env(monkeypatch, tmp_path)
    center = np.full(track.len - 1, 0.5)  # circular track: one position per border pair

    accepted, improved = accept_result(track, car, center)

    assert improved is True
    assert np.isfinite(accepted.best_time)
    stored, found = registry.load_result(track, car)
    assert found
    stored.simulate(car)  # load_result restores the line, not best_time
    assert stored.best_time == pytest.approx(accepted.best_time, rel=1e-9)


def test_accept_result_keeps_the_faster_line(monkeypatch, tmp_path):
    from laptime_sim import registry

    track, car = _synthetic_env(monkeypatch, tmp_path)
    idx_len = track.len - 1
    center = np.full(idx_len, 0.5)
    inner = BORDER_CLEARANCE_M / Raceline(track).width + 0.01

    first, _ = accept_result(track, car, center)
    second, _ = accept_result(track, car, inner)
    (fast_time, fast_pos), (slow_time, slow_pos) = sorted([(first.best_time, center), (second.best_time, inner)])
    assert slow_time > fast_time

    _, improved = accept_result(track, car, slow_pos)

    assert improved is False
    stored, _ = registry.load_result(track, car)
    stored.simulate(car)
    assert stored.best_time == pytest.approx(fast_time, rel=1e-9)


def test_accept_result_rejects_invalid_positions(monkeypatch, tmp_path):
    track, car = _synthetic_env(monkeypatch, tmp_path)
    idx_len = track.len - 1

    with pytest.raises(ValueError, match="shape"):
        accept_result(track, car, np.full(idx_len + 1, 0.5))
    with pytest.raises(ValueError, match="non-finite"):
        accept_result(track, car, np.where(np.arange(idx_len) == 0, np.nan, 0.5))
    with pytest.raises(ValueError, match="clearance"):
        accept_result(track, car, np.full(idx_len, 1.5))
