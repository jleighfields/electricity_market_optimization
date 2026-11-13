# LP in electricity markets presentations

Two slide decks and a standalone demo, all [marimo](https://marimo.io) notebooks, with live LP/MILP demos
(PyPSA + HiGHS, SciPy `linprog`). Contributor conventions (Claude Code included) are in `CLAUDE.md`.

## Repository layout

| Path | What |
|:---|:---|
| `lp_electricity_markets.py` | Main deck: short-term markets |
| `lp_planning.py` | Companion deck (work in progress): mid- and long-term planning |
| `optimality_gap_demo.py` | Standalone notebook, not a deck: watch a unit commitment MILP's optimality gap close |
| `*.html` | Static exports of the notebooks above (see below) |
| `layouts/` | marimo slides layout for each deck |
| `assets/` | All data, so the talks run offline (table below), plus the two small widget scripts (`theme_toggle.js`, `map_fade.js`); `assets/raw/` is git-ignored |
| `scripts/build_data.py` | Downloads and rebuilds the data in `assets/` |
| `scripts/capture_ercot_map.py` | Captures the ERCOT LMP map used on the slides |
| `scripts/build_gap_trace.py` | Re-runs `optimality_gap_demo.py` and saves its trace for the main deck |
| `CLAUDE.md` | Conventions and the after-every-change checklist |
| `app.py`, `requirements.txt` | Serve all three notebooks as one web app (Posit Connect or any ASGI host; see below) |

## `lp_electricity_markets.py`: short-term markets

- maps of US plants, ISOs/RTOs, WECC balancing authorities and the western grid, and how the map becomes buses and lines
- merit order and economic dispatch as an LP; a 24-hour day with a duck curve, ramp limits and a battery
- two-bus congestion, three-bus loop flow and negative prices (DC-OPF)
- DC-OPF prices on a 5,076-bus western grid model
- co-optimized energy and reserves
- 24-hour unit commitment MILP, the convex-hull view of why MILPs lack duals, uplift, and the optimality gap
- LSE portfolio hedging on real CAISO prices: CVaR-limited hedging LP with fixed-price contracts, price sensitivity,
  and the highest acceptable price for bilateral offers, read from the solved LP
- glossary, sources and live ISO price-map links in the appendix

## `optimality_gap_demo.py`: unit commitment and the optimality gap

A standalone notebook (not part of the slides): builds a unit commitment MILP (10–80 units, 24–72 hours),
solves it with HiGHS, and plots the best solution and best bound over time, so you can watch the gap close.
Run it with `uv run marimo run optimality_gap_demo.py`; pick the size, time limit and target gap, then press **Solve**.

## `lp_planning.py`: mid- and long-term planning (work in progress)

- the missing-money problem, resource adequacy approaches, and the ISO's capacity demand curve
- capacity auction clearing (the price is the dual of the balance)
- integrated resource planning: a PyPSA capacity-expansion LP on two representative days, with a
  planning reserve margin and a CO2 cap whose duals are the capacity and carbon prices
- zero profit: every built resource's energy margin plus capacity payment equals its build cost

## Run it on any computer

1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
2. From this folder:

```bash
uv sync                                           # builds .venv from uv.lock (Python 3.12)
uv run marimo run lp_electricity_markets.py       # present: opens in slides layout
uv run marimo edit lp_electricity_markets.py      # edit; toggle slides view in the app-view menu
uv run marimo run lp_planning.py                  # the planning deck
```

## Static HTML backup

```bash
uv run marimo export html lp_electricity_markets.py -o lp_electricity_markets.html
uv run marimo export html lp_planning.py -o lp_planning.html
uv run marimo export html optimality_gap_demo.py -o optimality_gap_demo.html
```

This produces one self-contained file (~8 MB) that opens in any browser with no Python.
The maps keep zoom, pan and hover. The sliders are frozen at their defaults,
because there's no kernel to re-solve the LPs.

## Hosting the live notebooks (Posit Connect)

`app.py` wraps the notebooks in one FastAPI app with marimo's `create_asgi_app()`, so sliders and switches
re-solve on the server:

| Path | Notebook |
|:---|:---|
| `/` | `lp_electricity_markets.py` (slides) |
| `/planning` | `lp_planning.py` (slides) |
| `/gap-demo` | `optimality_gap_demo.py` |

Try it locally first:

```bash
uv run --group connect uvicorn app:app --port 8080    # then open http://localhost:8080
```

Deploy with [rsconnect-python](https://docs.posit.co/rsconnect-python/) (after `rsconnect add` for your server):

```bash
rsconnect deploy fastapi --entrypoint app:app \
    --exclude ".venv/*" --exclude "assets/raw/*" --exclude "__marimo__/*" --exclude "*.html" .
```

- **Posit Connect Cloud (deploy from GitHub):** choose framework **Shiny**, primary file `app.py`, and **Python 3.12** or newer
  (the default 3.11 is too old for the pinned packages). Connect Cloud has no FastAPI option, but `shiny run app:app`
  serves any ASGI app, so `shiny` is in `requirements.txt`.
- Connect installs `requirements.txt`. After changing dependencies, regenerate it:
  `uv export --group connect --no-hashes --no-dev --no-emit-project --format requirements-txt -o requirements.txt`
- marimo needs WebSockets between the browser and the server; Connect must allow them (it does for Shiny apps).
- Ship `assets/` (stored solves) and `layouts/` (slides layout). Each visitor gets their own Python session,
  and the gap demo solves its MILP (~30 s) when it opens.

## Data (all stored in `assets/`, so the talk runs offline; only the map background tiles need internet)

| File | What | Source |
|---|---|---|
| `plants.parquet` | 14,063 operating plants (EIA-860, 2025) | PUDL, Catalyst Cooperative |
| `us_states.geojson` | Lower-48 state outlines | US Census cartographic boundaries |
| `western_16h_model.json` | 5,076-bus western grid model | Microsoft GridSFM (Hugging Face, MIT) |
| `western_opf_*.parquet`, `western_opf_timing.json` | Stored DC-OPF result (LMPs, line flows, path prices) and its solve time | Written by the notebook on first run |
| `ercot_best*.png`, `ercot_best.json` | ERCOT real-time LMP map (most congested interval captured) and its prices | ercot.com, via `scripts/capture_ercot_map.py` |
| `ercot_rtm_lmp_*.png` | Fallback ERCOT LMP map | ercot.com |
| `uc_gap_trace.parquet`, `uc_gap_result.json` | Optimality-gap convergence trace for the main deck | `scripts/build_gap_trace.py` |
| `pge_hourly.parquet` | Hourly PG&E DLAP day-ahead price (with energy/congestion/loss parts), PG&E-area load, CAISO solar; Jan 2024 – Oct 2026 | CAISO OASIS; EIA-930 via PUDL |
| `raw/caiso_oasis/` | Monthly OASIS downloads behind `pge_hourly.parquet` | CAISO OASIS |

To refresh the downloaded data: `uv run python scripts/build_data.py` (or one step:
`plants`, `states`, `grid`, `pge`). OASIS only keeps about three years of prices and
throttles bursts of requests, so the `pge` step pauses between months (~15 min) and
resumes from `raw/caiso_oasis/` if interrupted.
To re-solve the western OPF (~40 s), delete `assets/western_opf_*.parquet` and `western_opf_timing.json`, then rerun the notebook.
To refresh the gap trace (~30 s): `uv run python scripts/build_gap_trace.py`.

The ERCOT bar chart is a fixed snapshot (interval ending 04:45, 11 Nov 2026, the same interval as the map).

## Further reading

- J. A. Taylor, *Convex Optimization of Power Systems*, Cambridge University Press, 2015.
- D. S. Kirschen and G. Strbac, *Fundamentals of Power System Economics*, 2nd ed., Wiley, 2018.
