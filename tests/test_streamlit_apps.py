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
