"""Serve the marimo notebooks as one FastAPI app, for Posit Connect (or any ASGI host).

    /           the short-term markets deck (lp_electricity_markets.py)
    /planning   the mid/long-term planning deck (lp_planning.py)
    /gap-demo   the optimality-gap notebook (optimality_gap_demo.py)

Run locally:   uv run --group connect uvicorn app:app --port 8080
Deploy:        rsconnect deploy fastapi --entrypoint app:app . (see README)
"""

from pathlib import Path

# Import the notebooks' heavy libraries once, here, before any session starts. marimo runs every
# viewer's notebook as a thread in this process; two sessions importing anywidget -> ipywidgets ->
# IPython at the same moment can deadlock on IPython's circular imports and leave it half-loaded,
# breaking every later session ("module 'IPython' has no attribute 'display").
import anywidget  # noqa: F401
import highspy  # noqa: F401
import IPython.display  # noqa: F401
import linopy  # noqa: F401
import numpy  # noqa: F401
import pandas  # noqa: F401
import plotly.graph_objects  # noqa: F401
import plotly.subplots  # noqa: F401
import pypsa  # noqa: F401
import scipy.optimize  # noqa: F401
import xarray  # noqa: F401

import marimo
from fastapi import FastAPI

HERE = Path(__file__).parent

server = (
    marimo.create_asgi_app()  # read-only "marimo run" apps, each with its saved layout (slides)
    .with_app(path="/planning", root=str(HERE / "lp_planning.py"))
    .with_app(path="/gap-demo", root=str(HERE / "optimality_gap_demo.py"))
    .with_app(path="", root=str(HERE / "lp_electricity_markets.py"))
)

app = FastAPI()
app.mount("/", server.build())
