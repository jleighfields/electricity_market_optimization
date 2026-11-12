import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium", app_title="Optimality Gap Demo")


@app.cell(hide_code=True)
def _():
    import time
    import warnings

    import highspy
    import linopy
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import xarray as xr
    from plotly.subplots import make_subplots

    warnings.filterwarnings("ignore")
    return go, highspy, linopy, make_subplots, mo, np, pd, time, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Unit commitment and the optimality gap

    A companion to *How electricity markets run on linear programming*. Unit commitment (which plants are **on** in
    each hour) is a **MILP**, and a day-ahead market must clear it before a deadline. So solvers run
    **branch and bound** with a time limit and stop at a target **optimality gap**.

    While it searches, the solver tracks two numbers:

    - **Best solution (incumbent):** the cheapest feasible schedule found so far, an **upper bound** on the optimal cost.
    - **Best bound:** a proven **lower bound**, from the LP relaxations of the open branch-and-bound nodes.
      No schedule can cost less than this.

    $$\text{gap} = \frac{\text{best solution} - \text{best bound}}{\text{best solution}}$$

    The solver stops when the gap falls below the target (the answer is then provably within that % of optimal),
    or when time runs out (it returns the best solution so far, **with** a guarantee of how far from optimal it can be).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    controls = mo.ui.batch(
        mo.md(
            """
    | | |
    |:---|:---|
    | {units} | {hours} |
    | {time_limit} | {target_gap} |
    """
        ),
        {
            "units": mo.ui.slider(10, 80, step=5, value=40, label="Generating units", show_value=True),
            "hours": mo.ui.dropdown(options=["24", "48", "72"], value="48", label="Hours"),
            "time_limit": mo.ui.slider(5, 120, step=5, value=30, label="Time limit (s)", show_value=True),
            "target_gap": mo.ui.dropdown(options=["1%", "0.1%", "0.01%"], value="0.01%", label="Target gap"),
        },
    ).form(submit_button_label="Solve")
    controls
    return (controls,)


@app.cell
def _(linopy, np, pd, xr):
    def build_unit_commitment(n_units, n_hours, seed=0):
        """A 3-binary unit commitment MILP: on/off u, startup v, shutdown w, output p, for every unit and hour."""
        rng = np.random.default_rng(seed)
        kind = rng.choice(["coal", "cc", "ct"], size=n_units, p=[0.25, 0.45, 0.30])
        pick = lambda coal, cc, ct: np.where(kind == "coal", coal, np.where(kind == "cc", cc, ct))
        fleet = pd.DataFrame({
            "kind": kind,
            "pmax": pick(rng.uniform(300, 500, n_units), rng.uniform(200, 400, n_units), rng.uniform(50, 150, n_units)).round(),
            "marginal_cost": pick(rng.uniform(20, 28, n_units), rng.uniform(30, 40, n_units), rng.uniform(60, 90, n_units)).round(1),
        })
        fleet["pmin"] = (fleet.pmax * pick(0.4, 0.3, 0.2)).round()            # can't run below this when on
        fleet["no_load_cost"] = (fleet.pmax * pick(8, 6, 4)).round()           # $/h just for being on
        fleet["startup_cost"] = (fleet.pmax * pick(120, 60, 15)).round()       # $ per start
        fleet["min_up"] = pick(8, 4, 1)                                        # hours: once on, stay on
        fleet["ramp"] = (fleet.pmax * pick(0.3, 0.5, 1.0)).round()             # MW per hour

        hour = np.arange(n_hours) % 24
        demand = fleet.pmax.sum() * (0.45 + 0.2 * np.sin((hour - 10) * np.pi / 12)
                                     + 0.08 * np.exp(-(((hour - 19) / 2.5) ** 2)))
        reserve = 0.07 * demand                                               # spare MW to hold online

        g, t = pd.RangeIndex(n_units, name="g"), pd.RangeIndex(n_hours, name="t")
        per_unit = lambda col: xr.DataArray(fleet[col].to_numpy(), coords=[g])
        per_hour = lambda values: xr.DataArray(values, coords=[t])

        m = linopy.Model()
        u = m.add_variables(binary=True, coords=[g, t], name="on")
        v = m.add_variables(binary=True, coords=[g, t], name="start")
        w = m.add_variables(binary=True, coords=[g, t], name="stop")
        p = m.add_variables(lower=0, coords=[g, t], name="output")

        m.add_constraints(p.sum("g") == per_hour(demand), name="balance")                 # supply = demand
        m.add_constraints(p <= per_unit("pmax") * u, name="max_output")                   # off → 0 MW
        m.add_constraints(p >= per_unit("pmin") * u, name="min_output")
        m.add_constraints((per_unit("pmax") * u - p).sum("g") >= per_hour(reserve), name="reserve")
        m.add_constraints((u - u.shift(t=1) - v + w).sel(t=slice(1, None)) == 0, name="on_off_logic")
        m.add_constraints((p - p.shift(t=1) - per_unit("ramp") - per_unit("pmax") * v).sel(t=slice(1, None)) <= 0,
                          name="ramp_up")
        m.add_constraints((p.shift(t=1) - p - per_unit("ramp") - per_unit("pmax") * w).sel(t=slice(1, None)) <= 0,
                          name="ramp_down")
        for i, unit in fleet[fleet.min_up > 1].iterrows():                  # started in the last L hours → on now
            for hr in range(n_hours):
                window = slice(max(0, hr - unit.min_up + 1), hr)
                m.add_constraints(v.loc[i, window].sum() <= u.loc[i, hr], name=f"min_up_{i}_{hr}")
                m.add_constraints(w.loc[i, window].sum() <= 1 - u.loc[i, hr], name=f"min_down_{i}_{hr}")

        m.add_objective((per_unit("marginal_cost") * p).sum() + (per_unit("no_load_cost") * u).sum()
                        + (per_unit("startup_cost") * v).sum())
        return m, fleet

    return (build_unit_commitment,)


@app.cell
def _(highspy, pd, time):
    def solve_with_trace(model, time_limit, target_gap):
        """Solve with HiGHS, recording the best solution and best bound every time the solver reports progress."""
        h = model.to_highspy()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("time_limit", float(time_limit))
        h.setOptionValue("mip_rel_gap", float(target_gap))
        trace = []

        def record(event):
            d = event.data_out
            trace.append((d.running_time, d.mip_primal_bound, d.mip_dual_bound, d.mip_node_count))

        h.cbMipInterrupt.subscribe(record)            # called often during branch and bound
        h.cbMipImprovingSolution.subscribe(record)    # called whenever a better schedule is found
        start = time.time()
        h.run()
        info = h.getInfo()
        trace.append((time.time() - start, info.objective_function_value, info.mip_dual_bound, info.mip_node_count))
        trace = pd.DataFrame(trace, columns=["seconds", "best_solution", "best_bound", "nodes"])
        trace = trace[trace.best_solution.abs() < 1e20]                   # before the first solution: no gap yet
        trace["gap"] = (trace.best_solution - trace.best_bound) / trace.best_solution.abs()
        return trace, {"status": h.modelStatusToString(h.getModelStatus()), "gap": info.mip_gap,
                       "nodes": info.mip_node_count, "seconds": time.time() - start,
                       "cost": info.objective_function_value, "bound": info.mip_dual_bound}

    return (solve_with_trace,)


@app.cell
def _(build_unit_commitment, controls, solve_with_trace):
    settings = controls.value or {"units": 40, "hours": "48", "time_limit": 30, "target_gap": "0.01%"}
    model, fleet = build_unit_commitment(settings["units"], int(settings["hours"]))
    gap_target = float(settings["target_gap"].rstrip("%")) / 100
    trace, result = solve_with_trace(model, settings["time_limit"], gap_target)
    return fleet, gap_target, model, result, settings, trace


@app.cell(hide_code=True)
def _(fleet, gap_target, go, make_subplots, mo, model, result, settings, trace):
    def gap_report():
        fig = make_subplots(2, 1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.08)
        fig.add_scatter(x=trace.seconds, y=trace.best_solution / 1e6, name="Best solution (upper bound)",
                        line=dict(shape="hv", color="#E07A1F", width=2.5), row=1, col=1)
        fig.add_scatter(x=trace.seconds, y=trace.best_bound / 1e6, name="Best bound (lower bound)",
                        line=dict(shape="hv", color="#2F4A7A", width=2.5), fill="tonexty",
                        fillcolor="rgba(160,160,160,0.25)", row=1, col=1)
        fig.add_scatter(x=trace.seconds, y=100 * trace.gap, name="Gap", showlegend=False,
                        line=dict(shape="hv", color="#6A6A6A", width=2), row=2, col=1)
        fig.add_hline(100 * gap_target, line=dict(color="#A84F0A", dash="dash"), row=2, col=1,
                      annotation_text=f"target {settings['target_gap']}", annotation_position="top right")
        fig.update_yaxes(title="Cost ($M)", row=1, col=1)
        fig.update_yaxes(title="Gap (%)", type="log", row=2, col=1)
        fig.update_xaxes(title="Seconds", row=2, col=1)
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), legend=dict(x=0.55, y=0.98),
                          plot_bgcolor="#F7F5EF", paper_bgcolor="#F7F5EF")

        n_binary = 3 * len(fleet) * int(settings["hours"])                  # on, start, stop per unit-hour
        dollars = result["cost"] - result["bound"]
        stopped = ("it reached the **target gap**" if result["gap"] <= gap_target + 1e-12
                   else "it hit the **time limit**")
        return mo.vstack([
            mo.md(
                f"**Size:** {len(fleet)} units × {settings['hours']} hours = **{model.nvars:,} variables** "
                f"({n_binary:,} binary) and **{model.ncons:,} constraints**."
            ),
            mo.ui.plotly(fig),
            mo.md(
                f"HiGHS stopped after **{result['seconds']:.1f} s** and {result['nodes']:,.0f} branch-and-bound nodes, "
                f"because {stopped}. Best schedule: **\\${result['cost']:,.0f}**; no schedule can cost less than "
                f"**\\${result['bound']:,.0f}**. So it is within **{100 * result['gap']:.3f}%** "
                f"(\\${dollars:,.0f}) of optimal.\n\n"
                "- The **first** feasible schedule usually arrives fast but loose; heuristics then improve it in big jumps.\n"
                "- The **bound** creeps up slowly: proving optimality is usually the hard part, not finding a good schedule.\n"
                "- Try more units or hours: the problem grows linearly, but the search can grow much faster."
            ),
        ])

    gap_report()
    return


if __name__ == "__main__":
    app.run()
