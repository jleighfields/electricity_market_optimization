"""Serve the marimo notebooks as one FastAPI app, for Posit Connect (or any ASGI host).

    /           the short-term markets deck (lp_electricity_markets.py)
    /planning   the mid/long-term planning deck (lp_planning.py)
    /gap-demo   the optimality-gap notebook (optimality_gap_demo.py)

Run locally:   uv run --group connect uvicorn app:app --port 8080
Deploy:        rsconnect deploy fastapi --entrypoint app:app . (see README)
"""

from pathlib import Path

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
