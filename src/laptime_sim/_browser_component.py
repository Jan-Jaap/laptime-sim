"""Declares the raceline-optimizer browser component.

`components.declare_component` resolves its calling module via
`inspect.getmodule`, which requires the module to be in `sys.modules`.
Streamlit pages are exec'd, not imported, so they are not in `sys.modules`
and declaring from a page raises `RuntimeError: module is None`. The
declaration therefore lives in this importable module.

The component's frontend assets (index.html/main.js/worker.js) live in
`src/streamlit_apps/components/raceline_optimizer/` — served by streamlit
from that path.
"""

from pathlib import Path

import streamlit.components.v1 as st_components

_COMPONENT_DIR = Path(__file__).resolve().parents[1] / "streamlit_apps" / "components" / "raceline_optimizer"
_component = None


def get_raceline_optimizer():
    """Returns the declared component, declaring it once per process."""
    global _component
    if _component is None:
        _component = st_components.declare_component("raceline_optimizer", path=str(_COMPONENT_DIR))
    return _component
