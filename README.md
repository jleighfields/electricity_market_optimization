# LP in electricity markets presentations

Two marimo notebooks that present as slides, with live LP/MILP demos (PyPSA + HiGHS, SciPy `linprog`).

## `lp_electricity_markets.py`: short-term markets


- merit order and single-bus economic dispatch
- two-bus congestion and three-bus loop flow (DC-OPF)
- maps of US plants, ISOs/RTOs, WECC balancing authorities and the western grid
- DC-OPF prices on a 5,076-bus western grid model
- co-optimized energy and reserves
- 24-hour unit commitment MILP, the convex-hull view of why MILPs lack duals, and uplift
- LSE portfolio hedging on real CAISO prices: CVaR-limited hedging LP with fixed-price contracts, price sensitivity,
  and the highest acceptable price for bilateral offers, read from the solved LP
- glossary, sources and live ISO price-map links in the appendix

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
```

This produces one self-contained file (~8 MB) that opens in any browser with no Python.
The maps keep zoom, pan and hover. The sliders are frozen at their defaults,
because there's no kernel to re-solve the LPs.

## Data (all stored in `assets/`, so the talk runs offline)

| File | What | Source |
|---|---|---|
| `plants.parquet` | 14,063 operating plants (EIA-860, 2025) | PUDL, Catalyst Cooperative |
| `us_states.geojson` | Lower-48 state outlines | US Census cartographic boundaries |
| `western_16h_model.json` | 5,076-bus western grid model | Microsoft GridSFM (Hugging Face, MIT) |
| `western_opf_*.parquet` | Stored DC-OPF result (LMPs, line flows) | Written by the notebook on first run |
| `ercot_rtm_lmp_*.png` | Fallback ERCOT LMP map | ercot.com (the slide fetches a live one when online) |
| `pge_hourly.parquet` | Hourly PG&E DLAP day-ahead price (with energy/congestion/loss parts), PG&E-area load, CAISO solar; Jan 2024 – Oct 2026 | CAISO OASIS; EIA-930 via PUDL |
| `raw/caiso_oasis/` | Monthly OASIS downloads behind `pge_hourly.parquet` | CAISO OASIS |

To refresh the downloaded data: `uv run python scripts/build_data.py` (or one step:
`plants`, `states`, `grid`, `pge`). OASIS only keeps about three years of prices and
throttles bursts of requests, so the `pge` step pauses between months (~15 min) and
resumes from `raw/caiso_oasis/` if interrupted.
To re-solve the western OPF (~40 s), delete `assets/western_opf_*.parquet` and rerun the notebook.

The ERCOT bar chart is a fixed snapshot (interval ending 01:45, 10 Nov 2026).

## Further reading

- J. A. Taylor, *Convex Optimization of Power Systems*, Cambridge University Press, 2015.
- D. S. Kirschen and G. Strbac, *Fundamentals of Power System Economics*, 2nd ed., Wiley, 2018.
