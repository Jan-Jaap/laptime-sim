import importlib.util
from pathlib import Path

STREAMLIT_PAGES = sorted(Path("src/streamlit_apps").rglob("*.py"))


def page_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_streamlit_pages_import():
    assert STREAMLIT_PAGES
    for page in STREAMLIT_PAGES:
        page_module(page)


def test_raceline_optimizer_component_declares_headlessly():
    """declare_component resolves its caller via inspect.getmodule, which fails
    for exec'd streamlit pages (they are not in sys.modules) — the declaration
    must live in the importable `laptime_sim._browser_component` module, not in
    a page."""
    from laptime_sim._browser_component import get_raceline_optimizer

    component = get_raceline_optimizer()

    assert component is not None
    assert component.name.endswith("raceline_optimizer")


def test_page4_headless_flow_and_track_switch():
    """Single AppTest instance covering the page lifecycle — AppTest page runs
    are expensive (real track loads + simulates), so everything shares one run.

    The optimization itself never runs in tests: the browser component has no
    worker under AppTest, so only the page's session-state handling is exercised."""
    from streamlit.testing.v1 import AppTest

    page = Path(__file__).resolve().parents[1] / "src" / "streamlit_apps" / "pages" / "4_racetrack_optimization.py"
    at = AppTest.from_file(page, default_timeout=120)
    at.run()
    assert not at.exception
    assert at.session_state["optimization_running"] is False

    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["optimization_running"] is True
    assert at.session_state["run_id"] == 2

    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["optimization_running"] is False
    # run_id identifies a run: stopping must not invalidate the in-flight result
    assert at.session_state["run_id"] == 2

    # regression: the component keeps its last value across reruns — switching
    # tracks must stop the optimization, bump run_id and clear the processed
    # marker, or the previous track's result is accepted against the new track
    # (border-clearance ValueError crashed the page)
    first_name = at.radio[0].value.name
    other_label = next(o for o in at.radio[0].options if o != first_name)
    at.radio[0].set_value(other_label).run()
    assert not at.exception

    assert at.session_state["optimization_running"] is False
    assert at.session_state["run_id"] == 3  # bumped past the stale result's run_id
    assert at.session_state["processed_result"] is None
    assert at.session_state["raceline_key"] == (other_label, at.radio[1].value.name)
