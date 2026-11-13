import marimo

__generated_with = "0.17.0"
app = marimo.App(
    width="medium",
    app_title="Optimization for Electricity Planning",
    layout_file="layouts/lp_planning.slides.json",
)


@app.cell(hide_code=True)
def _():
    import logging
    import warnings

    from pathlib import Path

    import anywidget
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import pypsa
    from scipy.optimize import linprog

    warnings.filterwarnings("ignore")
    logging.getLogger("pypsa").setLevel(logging.ERROR)
    logging.getLogger("linopy").setLevel(logging.ERROR)

    QUIET = {"output_flag": False}  # silence the HiGHS log on every solve

    class _ThemeToggle(anywidget.AnyWidget):
        """A button on the title slide that flips the deck between light and dark mode."""
        _esm = Path(__file__).parent / "assets" / "theme_toggle.js"

    def terms(text):
        """A small line that defines the jargon and symbols used on a slide."""
        return mo.md(f"<small>**Terms** — {text}</small>")

    def show_chart(fig, height, **layout):
        """Show a Plotly chart at a fixed height so the slide fits."""
        fig.update_layout(
            height=height, margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="#F7F5EF", paper_bgcolor="#F7F5EF", font=dict(size=13),
            legend=dict(bgcolor="#F7F5EF", traceorder="normal"),
        )
        fig.update_layout(**layout)
        axis = dict(showline=True, linecolor="#6A6A6A", ticks="outside", showgrid=False, zeroline=False)
        fig.update_xaxes(**axis)
        fig.update_yaxes(**axis)
        return mo.ui.plotly(fig, config={"displaylogo": False})

    requirement_slider = mo.ui.slider(
        8000, 11000, step=100, value=10000, label="Capacity requirement (MW)", show_value=True
    )
    co2_slider = mo.ui.slider(
        30, 100, step=10, value=60, label="CO₂ cap (% of uncapped emissions)", show_value=True
    )

    _title = mo.md(
        """
    # Planning ahead: optimization for mid- and long-term electricity markets

    ### Capacity markets, resource adequacy, and integrated resource planning

    A companion to *How electricity markets run on linear programming* (short-term markets). **Work in progress.**

    Code and data: [github.com/jleighfields/electricity_market_optimization](https://github.com/jleighfields/electricity_market_optimization)
    """
    )
    mo.vstack([_title, mo.ui.anywidget(_ThemeToggle())], align="center")
    return (
        QUIET,
        co2_slider,
        go,
        linprog,
        mo,
        np,
        pd,
        pypsa,
        requirement_slider,
        show_chart,
        terms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Outline

    | # | Section | What we’ll see | Optimization idea |
    |:---|:---|:---|:---|
    | 0 | [**From minutes to decades**](#/2) | Which decisions are made how far ahead, and what changes when the variable is MW *built* | — |
    | 1 | [**Capacity markets**](#/3) | Missing money, resource adequacy, the ISO’s demand curve, clearing an auction | LP; the capacity price is a dual |
    | 2 | [**Integrated resource planning**](#/9) | What to build for the next 20 years: a capacity-expansion LP with a reserve margin and a CO₂ cap | Duals price energy, capacity and carbon; zero profit |
    | 3 | [**Still to come**](#/16) | Topics to build out next | MILP, stochastic programs |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## From minutes to decades: who optimizes what

    | Horizon | Decision | Who | Optimization |
    |:---|:---|:---|:---|
    | **Seconds to a day** | Dispatch, unit commitment, reserves | ISO | LP / MILP every 5 min and day-ahead (*the short-term talk*) |
    | **Months to ~3 years** | Hedges and contracts, fuel, hydro and maintenance schedules, **capacity auctions and RA** | ISO, utilities, LSEs, generators | LP / MILP over scenarios |
    | **5 to 30 years** | **What to build or retire**: generation, storage, transmission (**IRP**) | Utilities, regulators, planners | Capacity-expansion LP / MILP, often stochastic |

    **What changes as the horizon grows:**
    - The decision variable shifts from **MWh produced** to **MW built**: capital costs enter the objective.
    - One LP has to represent **many hours** (often a few *representative days*) and many uncertain futures.
    - The duals still mean something: a **capacity price** (\$ per MW-year) and a **carbon price** (\$ per ton) appear next to the energy price.

    <small>**Terms** — **RA**: resource adequacy, enough capacity for peak plus a margin &ensp;|&ensp; **LSE**: load-serving entity, a utility or retailer that buys power for customers &ensp;|&ensp; **IRP**: integrated resource plan, a utility’s long-term build plan, reviewed by its regulator</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 1 · Capacity markets

    **So far (short-term talk):** the ISO buys energy and reserves for the next few minutes or the next day.

    **But:** those prices may not pay for enough plants to exist when the hottest hour of a future year arrives.

    **Next:** buy **MW of capacity** directly, years ahead, in an auction that is itself an LP.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Why pay for capacity? The *missing money* problem

    Will enough plants exist for the hottest hour of a year three years from now? Energy prices alone may not get them built:

    - A **peaker** may run only a few hundred hours a year, and must earn back its fixed costs in those hours.
    - **Price and offer caps** clip exactly those scarcity hours (FERC caps ISO/RTO offers at \$1,000/MWh, or verified cost up to \$2,000).
    - **Demand barely responds**, so in a true shortage operators shed load instead of raising the price. Reliability is
      shared by everyone, so no single buyer pays for enough of it.

    | Response | Where | Idea |
    |:---|:---|:---|
    | **Pay separately for MW** | PJM, MISO, ISO-NE, NYISO; California (bilateral RA) | Capacity auctions or RA requirements |
    | **Let scarcity prices run high** | ERCOT (energy-only) | Prices up to \$5,000/MWh when reserves run short |

    <small>**Terms** — **peaker**: plant that runs only at peak demand &ensp;|&ensp; **fixed costs**: capital and upkeep, paid whether or not the plant runs &ensp;|&ensp; **shed load**: controlled blackouts &ensp;|&ensp; **resource adequacy (RA)**: enough capacity to meet peak plus a margin</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Paying for capacity: resource adequacy

    The first answer to *missing money*: pay for **MW** directly. Three US approaches:

    | Approach | Where | How it works |
    |:---|:---|:---|
    | **Centralized capacity auction** | PJM, MISO, ISO-NE, NYISO | The ISO buys capacity against a sloped demand curve, from weeks ahead (MISO, NYISO) to up to about 3 years ahead (PJM); one clearing price per zone. |
    | **Bilateral RA market** | California (CAISO) | No auction. LSEs buy RA bilaterally from resource owners, and the CPUC requires each of its LSEs to show contracts covering its load in **every hour of each month’s peak day** (“slice-of-day,” since 2025) plus a planning reserve margin. Publicly owned utilities follow their own regulators. CAISO enforces offer obligations and backstop procurement. |
    | **Energy-only** | ERCOT | No capacity product. Scarcity pricing: reserve demand curves, co-optimized with energy, raise prices when reserves run short. |

    In an **auction**, the **capacity price is a dual**: the shadow price of the reliability requirement. (A bilateral RA price is negotiated; ERCOT has no capacity price.)

    <small>**Terms** — **capacity** (the product): a promise to be available, paid per MW &ensp;|&ensp; **CPUC**: California’s utility regulator &ensp;|&ensp; **planning reserve margin**: required capacity above forecast peak &ensp;|&ensp; **bilateral**: negotiated buyer-to-seller, not cleared by the ISO &ensp;|&ensp; **offer obligation**: RA capacity must be offered into CAISO &ensp;|&ensp; **backstop procurement**: CAISO buys any shortfall &ensp;|&ensp; **scarcity pricing**: very high prices when reserves run short</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Where the demand curve comes from

    In a capacity auction, consumers don’t bid. **The ISO sets the demand curve** from two studies:

    | Axis | Set by | How |
    |:---|:---|:---|
    | **Quantity** (MW) | Reliability study | Forecast peak load plus the reserve margin needed to meet a loss-of-load target, typically **“one day in ten years.”** This is the *reliability requirement*. |
    | **Price** (\$/MW-day) | Net CONE | **Cost of new entry** for a reference new plant (usually a gas combustion turbine, as in PJM), **net** of the energy and reserve revenue it expects to earn: the “missing money” a new plant needs. |

    - The curve **slopes down** because each MW beyond the requirement buys less extra reliability, and a
      vertical requirement would make prices jump between zero and the cap.
    - PJM calls it the **VRR (Variable Resource Requirement) curve**; MISO, ISO-NE and NYISO have their own versions.
    - In the LP, each demand-curve step $m$ is a variable $d_m$ (MW bought) worth $w_m$ (\$/MW-day), so the auction **maximizes welfare**:
      the area under the demand curve minus the cost of the cleared offers.

    <small>**Terms** — **\$/MW-day**: price to keep 1 MW available for one day &ensp;|&ensp; **LOLE** (loss-of-load expectation): expected days per year with too little supply; “one day in ten years” = 0.1 &ensp;|&ensp; **CT**: combustion turbine, a simple gas peaker</small>
    """
    )
    return


@app.cell
def _(terms, linprog, mo, np, pd):
    def clear_capacity_auction(requirement):
        offers = pd.DataFrame({  # $/MW-day
            "resource": ["Nuclear", "Gas CC", "Coal", "Battery", "Gas CT",
                         "Demand response", "New CT"],
            "mw": [3000, 3000, 2000, 800, 1500, 500, 1000],
            "price": [0, 30, 50, 90, 120, 200, 300],
        })
        net_cone = 250  # $/MW-day: a reference new plant's fixed cost minus its expected market revenue
        demand = pd.DataFrame({  # set by the ISO: price steps relative to Net CONE,
            "mw": [requirement - 600, 600, 600],  # centered on the reliability requirement
            "value": [1.8 * net_cone, net_cone, 0.32 * net_cone],
        })
        # Variables: x_i = MW cleared from offer i, d_m = MW bought on demand step m.
        # max welfare = min Σ c_i x_i − Σ w_m d_m  s.t.  Σ x_i = Σ d_m  (π),  bounds below
        cost = np.concatenate([offers.price, -demand.value])        # objective coefficients
        balance = np.concatenate([np.ones(len(offers)), -np.ones(len(demand))])  # Σx − Σd = 0
        bounds = [(0, q) for q in offers.mw] + [(0, q) for q in demand.mw]       # step sizes
        res = linprog(cost, A_eq=[balance], b_eq=[0], bounds=bounds, method="highs")
        offers["cleared"] = res.x[: len(offers)]
        price = res.eqlin.marginals[0]  # dual of the balance row: the clearing price π
        return offers, demand, price

    mo.vstack([mo.md("## Code: clearing a capacity auction with `scipy.optimize.linprog`"), mo.show_code(), terms(r"**demand response**: customers paid to cut use on request &ensp;|&ensp; $x_i$, $c_i$: MW cleared from offer $i$ and its price &ensp;|&ensp; $d_m$, $w_m$: MW bought on demand step $m$ and its value &ensp;|&ensp; $\pi$: clearing price (\$/MW-day)")])
    return (clear_capacity_auction,)


@app.cell(hide_code=True)
def _(clear_capacity_auction, go, mo, np, requirement_slider, show_chart):
    def capacity_auction_slide(requirement):
        offers, demand, price = clear_capacity_auction(requirement)

        fig = go.Figure()
        supply = offers.sort_values("price")
        fig.add_trace(go.Scatter(
            x=np.concatenate([[0], supply.mw.cumsum()]),
            y=np.concatenate([supply.price, [supply.price.iloc[-1]]]),
            mode="lines", line=dict(shape="hv", color="#2F4A7A", width=2.5), name="Supply offers",
        ))
        fig.add_trace(go.Scatter(
            x=np.concatenate([[0], demand.mw.cumsum()]), y=np.concatenate([demand.value, [0]]),
            mode="lines", line=dict(shape="hv", color="#E07A1F", width=2.5), name="Demand curve",
        ))
        fig.add_hline(price, line=dict(color="#A84F0A", dash="dash", width=1.5))
        fig.add_annotation(x=200, y=price, text=f"<b>Clearing price ${price:.0f}/MW-day</b>",
                           showarrow=False, xanchor="left", yanchor="bottom", yshift=4,
                           font=dict(color="#A84F0A"))
        fig.add_vline(requirement, line=dict(color="#6A6A6A", dash="dot", width=1.5))
        side = 1 if requirement <= 10300 else -1  # label beside the line, away from the plot edge
        fig.add_annotation(x=requirement + side * 150, y=470, text="Reliability<br>requirement",
                           showarrow=False, xanchor="left" if side > 0 else "right", yanchor="top",
                           align="left" if side > 0 else "right", font=dict(size=12, color="#6A6A6A"))
        fig.update_xaxes(title="MW")
        fig.update_yaxes(title="$/MW-day", range=[0, 480])

        cleared = offers.set_index("resource")[["price", "mw", "cleared"]].round(0)
        return mo.vstack(
            [
                mo.md("## Clearing the auction: the price is the dual of the balance"),
                requirement_slider,
                mo.hstack([show_chart(fig, 380, legend=dict(x=0.01, y=0.5, yanchor="middle")), cleared],
                          widths=[1.4, 1]),
                mo.md(
                    f"Cleared **{offers.cleared.sum():,.0f} MW**. Every cleared resource is paid "
                    f"**\\${price:.0f}/MW-day**, the dual $\\pi$ of MW cleared = MW bought, whatever it offered. "
                    "Raise the requirement and watch more expensive resources clear and the price step up.\n\n"
                    "<small>Illustrative offers and demand curve: steps at 1.8×, 1.0× and 0.32× a Net CONE "
                    "of \\$250/MW-day, centered on the requirement.</small>"
                ),
            ]
        )

    capacity_auction_slide(requirement_slider.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 2 · Integrated resource planning

    **So far:** a capacity auction buys MW from plants that already exist or are already planned.

    **But:** someone has to decide *what* to build, for loads, fuel prices and policies 10–20+ years out.

    **Next:** a **capacity-expansion LP** chooses MW to build and how to run them, all at once.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## What is an integrated resource plan (IRP)?

    A utility’s long-term plan for **what to build, buy or retire** to serve its customers reliably at least cost. Most
    states require one every 2–5 years, reviewed in public by the state regulator.

    | Step | What happens | Optimization |
    |:---|:---|:---|
    | 1. **Forecast** | Load (electrification, data centers), fuel prices, technology costs, policy | Inputs and scenarios |
    | 2. **Resource options** | Solar, wind, storage, gas, efficiency, demand response, transmission, contracts | Candidate variables |
    | 3. **Capacity expansion** | Pick the least-cost mix that meets reliability and policy targets | **LP / MILP**, decades at once |
    | 4. **Production cost** | Re-run the chosen mix hour by hour, with unit commitment | **MILP** (the short-term talk) |
    | 5. **Risk analysis** | Test portfolios across load, price and policy scenarios | Stochastic or scenario LPs |
    | 6. **Preferred portfolio** | The plan the utility files; regulators and stakeholders weigh in | — |

    In the West outside CAISO, vertically integrated utilities (PacifiCorp, Xcel, APS, NV Energy, …) file IRPs; in California, the CPUC runs an IRP across its LSEs.

    <small>**Terms** — **vertically integrated**: one utility owns generation, transmission and retail service &ensp;|&ensp; **production cost model**: an hourly dispatch and commitment simulation of a fixed fleet &ensp;|&ensp; **demand response**: customers paid to cut use on request</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Capacity expansion is an LP: build and operate together

    $$
    \begin{aligned}
    \min\quad & \sum_g C_g K_g + \sum_t w_t \sum_g c_g\, p_{g,t} && & \color{#6A7179}\text{annualized build cost + yearly operating cost} \\
    \text{s.t.}\quad & \textstyle\sum_g p_{g,t} = D_t && (\lambda_t) & \color{#6A7179}\text{supply = demand, every hour} \\
    & 0 \le p_{g,t} \le a_{g,t}\, K_g && & \color{#6A7179}\text{run no more than you built (wind, solar: } a_{g,t} < 1\text{)} \\
    & \textstyle\sum_g \text{ELCC}_g\, K_g \ge (1+\text{PRM})\, \max_t D_t && (\kappa) & \color{#6A7179}\text{planning reserve margin} \\
    & \textstyle\sum_t w_t \sum_g e_g\, p_{g,t} \le E && (\mu) & \color{#6A7179}\text{CO}_2\text{ cap} \\
    & K_g \ge 0 && & \color{#6A7179}\text{(plus battery state of charge)}
    \end{aligned}
    $$

    - Same as economic dispatch, plus **capacity variables** $K_g$ and two **planning rows**. Each row’s dual is a price:
      $\lambda_t$ energy (\$/MWh), $\kappa$ **capacity** (\$/MW-yr), $\mu$ **carbon** (\$/t).
    - **Representative days** with weights $w_t$ stand in for the 8,760 hours of a year.
    - Real IRPs add build years, retirements, lumpy units (MILP), transmission and many scenarios.

    <small>**Terms** — $K_g$: MW of technology $g$ built &ensp;|&ensp; $C_g$: its annualized capital cost (\$/MW-yr) &ensp;|&ensp; $p_{g,t}, c_g$: output and operating cost &ensp;|&ensp; $a_{g,t}$: availability (wind/solar profile) &ensp;|&ensp; **ELCC**: effective load-carrying capability, the share of a MW that counts toward the peak &ensp;|&ensp; **PRM**: planning reserve margin &ensp;|&ensp; $e_g$: tons CO₂ per MWh &ensp;|&ensp; $E$: the cap</small>
    """
    )
    return


@app.cell
def _(mo, np, pd, terms):
    hours = pd.RangeIndex(48, name="hour")  # two representative days: a typical day, then a peak day
    _h = np.arange(24)
    _typical = 600 + 180*np.sin((_h-10)*np.pi/12) + 50*np.sin((_h-14)*np.pi/6)
    plan_demand = np.r_[_typical, 1.25*_typical]                        # MW
    plan_avail = {"Solar": np.tile(np.clip(np.sin((_h-6)*np.pi/12), 0, None), 2),
                  "Wind": np.r_[0.40 + 0.25*np.cos((_h-2)*np.pi/12), 0.5*(0.40 + 0.25*np.cos((_h-2)*np.pi/12))]}
    day_weight = np.r_[np.full(24, 330.0), np.full(24, 35.0)]          # days per year each one stands for
    tech = pd.DataFrame({  # capex: annualized $/MW-yr; mc: $/MWh; co2: t/MWh; elcc: share counted toward peak
        "Solar":   [ 70_000,  0, 0.00, 0.10],
        "Wind":    [130_000,  0, 0.00, 0.15],
        "Battery": [110_000,  0, 0.00, 0.90],  # 4-hour battery
        "Gas CC":  [130_000, 35, 0.37, 0.95],
        "Gas CT":  [ 85_000, 70, 0.55, 0.95],
    }, index=["capex", "mc", "co2", "elcc"]).T

    mo.vstack([mo.md("## Code (1/2): inputs for two representative days"), mo.show_code(), terms(
        r"`plan_demand`: $D_t$ &ensp;|&ensp; `plan_avail`: $a_{g,t}$ &ensp;|&ensp; `day_weight`: $w_t$ (330 typical days + 35 peak days = a year) &ensp;|&ensp; "
        r"`capex`: $C_g$ &ensp;|&ensp; `mc`: $c_g$ &ensp;|&ensp; `co2`: $e_g$ &ensp;|&ensp; `elcc`: ELCC$_g$")])
    return day_weight, hours, plan_avail, plan_demand, tech


@app.cell
def _(QUIET, day_weight, hours, mo, plan_avail, plan_demand, pypsa, tech, terms):
    def plan_build(co2_cap=None, prm=0.15):
        n = pypsa.Network()
        n.set_snapshots(hours)
        n.snapshot_weightings["objective"] = day_weight      # each hour counts for w_t hours per year
        n.snapshot_weightings["generators"] = day_weight
        n.add("Bus", "grid")
        n.add("Load", "load", bus="grid", p_set=plan_demand)
        for g, (capex, mc, co2, _) in tech.iterrows():
            n.add("Carrier", g, co2_emissions=co2)
            if g == "Battery":
                n.add("StorageUnit", g, bus="grid", carrier=g, p_nom_extendable=True, capital_cost=capex,
                      max_hours=4, efficiency_store=0.92, efficiency_dispatch=0.92, cyclic_state_of_charge=True)
            else:
                n.add("Generator", g, bus="grid", carrier=g, p_nom_extendable=True, capital_cost=capex,
                      marginal_cost=mc, p_max_pu=plan_avail.get(g, 1.0))
        if co2_cap is not None:  # Σ_t w_t Σ_g e_g p_gt ≤ E   (dual μ)
            n.add("GlobalConstraint", "co2", carrier_attribute="co2_emissions", sense="<=", constant=co2_cap)

        def reserve_margin(n, snapshots):  # Σ_g ELCC_g K_g ≥ (1 + PRM) · peak   (dual κ)
            K = n.model["Generator-p_nom"]
            lhs = sum(tech.elcc[g] * K.loc[g] for g in K.coords["name"].values)
            lhs = lhs + tech.elcc["Battery"] * n.model["StorageUnit-p_nom"].loc["Battery"]
            n.model.add_constraints(lhs >= (1 + prm) * plan_demand.max(), name="reserve_margin")

        n.optimize(solver_name="highs", solver_options=QUIET, extra_functionality=reserve_margin)
        return n

    mo.vstack([mo.md("## Code (2/2): capacity expansion as a PyPSA LP"), mo.show_code(), terms(
        r"`p_nom_extendable`: let the LP choose the MW built ($K_g$) &ensp;|&ensp; `capital_cost`: $C_g$ &ensp;|&ensp; `snapshot_weightings`: $w_t$ &ensp;|&ensp; "
        r"`GlobalConstraint`: the CO₂ cap &ensp;|&ensp; `extra_functionality`: adds our own row (the reserve margin) to the linopy model")])
    return (plan_build,)


@app.cell(hide_code=True)
def _(co2_slider, day_weight, go, mo, np, pd, plan_build, plan_demand, show_chart, tech):
    colors = {"Solar": "#E8C547", "Wind": "#5B9A6B", "Battery": "#3E8FB0", "Gas CC": "#2F4A7A", "Gas CT": "#E07A1F"}

    def emissions(n):
        return float((n.generators_t.p.mul(day_weight, axis=0).sum() * tech.co2.reindex(n.generators.index)).sum())

    uncapped = plan_build()
    uncapped_co2 = emissions(uncapped)

    def solve_plan(pct):
        return uncapped if pct >= 100 else plan_build(co2_cap=pct / 100 * uncapped_co2)

    def plan_slide(n):
        built = pd.concat([n.generators.p_nom_opt, n.storage_units.p_nom_opt]).reindex(tech.index)
        kappa = n.model.constraints["reserve_margin"].dual.item()
        mu = -n.global_constraints.mu.iloc[0] if len(n.global_constraints) else 0.0
        lam = n.buses_t.marginal_price["grid"]

        bars = go.Figure(go.Bar(x=built.index, y=built.values, marker_color=[colors[g] for g in built.index],
                                text=[f"{v:,.0f}" for v in built.values], textposition="outside", cliponaxis=False))
        bars.update_yaxes(title="MW built", range=[0, 1300])

        peak = slice(24, 48)  # the peak representative day
        h = np.arange(24)
        disp = n.generators_t.p.iloc[peak].copy()
        batt = n.storage_units_t.p["Battery"].iloc[peak]
        disp["Battery"] = batt.clip(lower=0)
        stack = go.Figure()
        for g in ["Wind", "Solar", "Gas CC", "Battery", "Gas CT"]:
            stack.add_trace(go.Scatter(x=h, y=disp[g].values, name=g, stackgroup="gen", line=dict(width=0, shape="hv"),
                                       fillcolor=colors[g], mode="lines"))
        stack.add_trace(go.Scatter(x=h, y=batt.clip(upper=0).values, name="Battery charging", fill="tozeroy",
                                   line=dict(width=0, shape="hv"), fillcolor="rgba(62,143,176,0.45)", mode="lines"))
        stack.add_trace(go.Scatter(x=h, y=plan_demand[peak], name="Demand", line=dict(color="black", width=2.5, shape="hv")))
        stack.update_xaxes(title="Hour of the peak day", dtick=3)
        stack.update_yaxes(title="MW")

        return mo.vstack([
            mo.md("## What the LP builds, and what its duals say"),
            co2_slider,
            mo.hstack([show_chart(bars, 330, showlegend=False),
                       show_chart(stack, 330, legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)))],
                      widths=[1, 1.6]),
            mo.md(
                f"Emissions **{emissions(n)/1e3:,.0f} kt CO₂/yr** · system cost **\\${n.objective/1e6:,.0f}M/yr** · "
                f"**capacity price** $\\kappa$ = **\\${kappa/1e3:,.0f}k/MW-yr** (\\${kappa/365:,.0f}/MW-day) · "
                f"**carbon price** $\\mu$ = **\\${mu:,.0f}/t** · peak-day energy price up to **\\${lam.iloc[peak].max():,.0f}/MWh**.\n\n"
                "Tighten the cap: wind, solar and batteries replace gas energy, but **gas CTs stay for the peak**, "
                "because they are the cheapest way to meet the reserve margin. Supply above the demand line charges the battery (below zero).\n\n"
                "<small>Illustrative costs and profiles: two representative days (typical × 330, peak × 35), 15% reserve margin.</small>"
            ),
        ])

    plan_net = solve_plan(co2_slider.value)
    plan_slide(plan_net)
    return (plan_net,)


@app.cell(hide_code=True)
def _(co2_slider, day_weight, mo, plan_net, tech):
    def zero_profit_slide(n, pct):
        lam = n.buses_t.marginal_price["grid"]
        kappa = n.model.constraints["reserve_margin"].dual.item()
        mu = -n.global_constraints.mu.iloc[0] if len(n.global_constraints) else 0.0
        rows = []
        for g in tech.index:
            if g == "Battery":
                K, energy = n.storage_units.p_nom_opt[g], (day_weight * lam * n.storage_units_t.p[g]).sum()
            else:
                p = n.generators_t.p[g]
                K = n.generators.p_nom_opt[g]
                energy = (day_weight * (lam - tech.mc[g] - mu * tech.co2[g]) * p).sum()
            if K > 1:
                rows.append({"Technology": f"**{g}**", "MW built": f"{K:,.0f}",
                             "Energy margin": f"${energy / K / 1e3:,.1f}k",
                             "+ Capacity (ELCC × κ)": f"${tech.elcc[g] * kappa / 1e3:,.1f}k",
                             "= Revenue": f"${(energy / K + tech.elcc[g] * kappa) / 1e3:,.1f}k",
                             "Build cost": f"${tech.capex[g] / 1e3:,.1f}k"})
        cols = list(rows[0])
        table = mo.md("\n".join(["| " + " | ".join(cols) + " |", "|:---|" + "---:|" * (len(cols) - 1)]
                                 + ["| " + " | ".join(str(r[c]).replace("$", "\\$") for c in cols) + " |" for r in rows]))
        return mo.vstack([
            mo.md(
                "## Every MW the LP builds exactly pays for itself\n\n"
                r"KKT stationarity for $K_g > 0$: $\;C_g = \sum_t w_t\,(\lambda_t - c_g - \mu e_g)\,\frac{p_{g,t}}{K_g} + \text{ELCC}_g\,\kappa$. "
                "Paid the LP’s prices, each built resource earns back its build cost, no more and no less (per MW-yr, "
                f"CO₂ cap at {pct}%):"
            ),
            table,
            mo.md(
                "- This is the **long-run equilibrium** a market is supposed to reach: no missing money, no excess profit.\n"
                "- The **capacity price** $\\kappa$ pays for the part of a peaker’s cost that energy prices don’t: the "
                "LP’s version of **Net CONE**, the number capacity auctions build their demand curves from.\n"
                "- A technology that is **not built** would earn less than its cost: its reduced cost is positive.\n\n"
                "<small>**Terms** — **energy margin**: $\\sum_t w_t(\\lambda_t - c_g - \\mu e_g)p_{g,t}$ per MW built (batteries: "
                "sell minus buy) &ensp;|&ensp; **Net CONE**: cost of new entry minus expected energy and reserve revenue</small>"
            ),
        ])

    zero_profit_slide(plan_net, co2_slider.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Still to come

    **Capacity markets and resource adequacy**
    - Where ELCC comes from: probabilistic reliability simulation (loss-of-load expectation) of the fleet.
    - PJM’s recent auctions: price spikes, price collars, and data-center load growth.
    - California’s slice-of-day RA: an hourly capacity requirement for each month’s peak day.

    **Integrated resource planning**
    - A real IRP (e.g., PacifiCorp or Xcel Colorado): inputs, candidate portfolios, the preferred plan.
    - Multi-year builds and retirements; **MILP** for lumpy units; linking to a production cost model.
    - Choosing **representative days** (time-series aggregation) and what it misses.
    - **Two-stage stochastic programs**: build now, operate in each scenario (see PyPSA’s
      [stochastic optimization example](https://docs.pypsa.org/latest/examples/stochastic-optimization/)).
    - Transmission expansion, and tools used in practice (PLEXOS, Aurora, EnCompass, GenX, PyPSA).

    **Other mid-term optimization:** hydro scheduling, maintenance outage scheduling, fuel contracts.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Sources

    | Topic | Source |
    |:---|:---|
    | Capacity markets | PJM, MISO, ISO-NE and NYISO capacity market manuals; FERC Order 831 (offer caps) |
    | California RA | CPUC Resource Adequacy proceeding (slice-of-day, from 2025) |
    | IRP | State commission IRP rules; utility IRP filings |
    | Capacity expansion | PyPSA documentation ([docs.pypsa.org](https://docs.pypsa.org)); J. D. Jenkins and N. A. Sepulveda, *GenX* |
    | Textbooks | D. S. Kirschen and G. Strbac, *Fundamentals of Power System Economics*, 2nd ed., Wiley, 2018 |

    <small>All costs, profiles and auction offers in this deck are illustrative, chosen to show how the LPs behave.</small>
    """
    )
    return


if __name__ == "__main__":
    app.run()
