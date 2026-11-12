# CLAUDE.md

Talks on LP/MILP in electricity markets for an LP class with no power-systems background.
Public repo: `jleighfields/electricity_market_optimization`.

See `README.md` for what each file is and how to run, export and rebuild everything.

## Stack

- **uv** for the environment (`uv sync`, `uv run ...`); add dependencies to `pyproject.toml`.
- **marimo** notebooks presented in slides layout: **one cell = one slide**.
- **Plotly** for every chart, through the deck's `show_chart(fig, height, ...)` helper (fixed height, `#F7F5EF` background, shared axis style).
- **PyPSA + HiGHS** for the power-system LPs/MILPs (`solver_options=QUIET`); custom rows via `extra_functionality` and linopy;
  **linopy + highspy** directly when solver callbacks are needed; **SciPy `linprog`** for small matrix-form LPs.

## Slide conventions

- Wrap per-slide logic in a named function (`def foo_slide(): ...` then `foo_slide()`), not `_`-prefixed cell variables.
- Every slide defines its jargon and symbols, usually in a `terms(...)` line at the bottom. Define constraints precisely
  (dispatchable $\underline P \le p \le \overline P$; wind/solar $0 \le p \le$ available, curtail only).
- Numbers on slides come from the code (solved models, data files), not typed in. Model slides show their size with `lp_size(n)`.
  Estimates and illustrative data are labeled as such.
- Plain words over jargon; the math is familiar to the audience, the physics is not.
- In f-strings, escape dollar signs as `\\$` so MathJax doesn't treat them as math.
- Avoid stacked `\frac` in slide text (the fraction bar can fail to render); write fractions inline.
- Colors: Solar `#E8C547`, Wind `#5B9A6B`, Nuclear `#8A6FB0`, Coal `#6A6A6A`, Gas CC `#2F4A7A`, Peaker `#E07A1F`,
  Battery `#3E8FB0`, price lines `#A84F0A`, binding limits `#E07A1F`.

## Expensive results are stored

- Western DC-OPF (~40 s): `assets/western_opf_*.parquet` + `western_opf_timing.json`. Delete them to re-solve. LP size is computed live.
- Gap trace (~30 s): `assets/uc_gap_trace.parquet` + `uc_gap_result.json`, rebuilt by `scripts/build_gap_trace.py`.

## After every change

1. `uv run marimo check <deck>.py` (only the existing markdown-indentation warnings are expected).
2. `uv run marimo export html <deck>.py -o <deck>.html`; the export must show no failed cells.
3. Check that every slide fits a 1294×853 viewport (no "Scroll for more") and look at changed slides.
4. Adding, removing or moving a slide shifts the outline links (`#/N`, 0-based slide index): update and test them.
5. Commit the `.py` and `.html` together and push.

Don't edit a notebook's `.py` while it's open in a live marimo session; edit through the session instead.

## Data gotchas

- CAISO OASIS: max 31 days per request, counted in Pacific time; keeps only ~3 years of prices; errors arrive as XML inside the zip.
- `gridstatus` breaks with pandas 3; call OASIS directly.
- California resource adequacy is bilateral, not an auction.
