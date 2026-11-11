import marimo

__generated_with = "0.17.0"
app = marimo.App(
    width="medium",
    app_title="LP in Electricity Markets",
    layout_file="layouts/lp_electricity_markets.slides.json",
)


@app.cell(hide_code=True)
def _():
    import itertools
    import json
    import logging
    import warnings
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import pypsa
    from plotly.subplots import make_subplots
    from scipy.optimize import linprog

    ASSETS = Path(__file__).parent / "assets"  # data saved by scripts/build_data.py

    warnings.filterwarnings("ignore")
    logging.getLogger("pypsa").setLevel(logging.ERROR)
    logging.getLogger("linopy").setLevel(logging.ERROR)

    QUIET = {"output_flag": False}  # silence the HiGHS log on every solve

    def terms(text):
        """A small line that defines the jargon and symbols used on a slide."""
        return mo.md(f"<small>**Terms** — {text}</small>")

    def show_chart(fig, height, **layout):
        """Show a Plotly chart styled like the maps, at a fixed height so the slide fits."""
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

    demand_slider = mo.ui.slider(
        100, 1000, step=10, value=820, label="Demand (MW)", show_value=True
    )
    limit_slider = mo.ui.slider(
        10, 200, step=10, value=100, label="Line limit (MW)", show_value=True
    )
    bus1_load_switch = mo.ui.switch(label="Add 1 MW of load at Bus 1")
    requirement_slider = mo.ui.slider(
        8000, 11000, step=100, value=10000, label="Capacity requirement (MW)", show_value=True
    )
    contract_dropdown = mo.ui.dropdown(
        options=["Flat forward", "On-peak forward", "Solar PPA"], value="Solar PPA", label="Contract"
    )
    cvar_slider = mo.ui.slider(  # feasible range for the PG&E scenarios (min CVaR ≈ $22.96M)
        23.0, 25.2, step=0.1, value=23.5, label="CVaR limit ($M)", show_value=True
    )

    mo.md(
        """
    # How electricity markets run on linear programming

    ### Merit order, locational prices, unit commitment, and portfolio management

    Code and data: [github.com/jleighfields/electricity_market_optimization](https://github.com/jleighfields/electricity_market_optimization)
    """
    ).center()
    return (
        ASSETS,
        QUIET,
        bus1_load_switch,
        demand_slider,
        go,
        itertools,
        json,
        limit_slider,
        linprog,
        make_subplots,
        mo,
        np,
        pd,
        pypsa,
        show_chart,
        contract_dropdown,
        cvar_slider,
        requirement_slider,
        terms,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Outline

    | | Section | What it covers | The optimization idea |
    |:---|:---|:---|:---|
    | 1 | [**Electricity markets 101**](#/2) | Why power is different, who’s involved, maps of plants, ISOs and the western grid, how bidding works | — |
    | 2 | [**Economic dispatch**](#/10) | The merit order, and why every generator is paid the same price | LP; the price **is** the dual |
    | 3 | [**Adding the network**](#/18) | Locational prices, congestion, loop flow, negative prices, the western grid, real ERCOT prices | Network flow + Kirchhoff’s law |
    | 4 | [**Reserves and capacity**](#/34) | Holding capacity in reserve, getting paid for energy and reserves, capacity markets | New constraints, new duals |
    | 5 | [**Unit commitment**](#/41) | Turning plants on and off a day ahead, and why that breaks pricing | MILP; no duals, uplift |
    | 6 | [**Portfolio management**](#/49) | An LSE hedging real CAISO prices: risk, contracts, what offers to accept | CVaR as an LP; reduced costs |
    | 7 | [**Wrap-up**](#/62) | Additional topics, a summary, glossary and sources | — |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Electricity is not like other goods

    | | |
    |:---|:---|
    | **Storage is scarce** | Supply must match demand *every second*, or frequency drifts and equipment trips. |
    | **Physics picks the path** | Power flows over every parallel line by Kirchhoff’s laws. You can’t route it like a package. |
    | **Demand barely responds** | Most consumers never see the real-time price, so demand is nearly fixed in the short run. |

    /// admonition | So…
    Someone has to decide, every few minutes, **who produces how much**, and that decision is an optimization problem.
    ///

    <small>**Terms** — **frequency**: the grid’s 60 Hz rhythm, which drifts when supply ≠ demand &ensp;|&ensp; **trip**: protective equipment disconnects automatically</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Who’s involved

    - **Generators**: gas, coal, nuclear, wind, solar, batteries. Each submits offers: *“up to X MW at \$Y/MWh.”*
    - **Load**: utilities and large customers buying power.
    - **Transmission**: the high-voltage network. Every line has a thermal limit.
    - **The ISO** (Independent System Operator): runs the market and **solves the optimization**.
      PJM, MISO, ERCOT, CAISO, NYISO, ISO‑NE, SPP.

    <small>**Terms** — **MW**: a rate of power &ensp;|&ensp; **MWh**: energy, 1 MW for 1 hour &ensp;|&ensp; **\$/MWh**: price of energy &ensp;|&ensp; **thermal limit**: most power a line carries before overheating &ensp;|&ensp; **RTO** (regional transmission organization): same role as an ISO</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(ASSETS, go, json, mo, pd):
    plants = pd.read_parquet(ASSETS / "plants.parquet")

    def load_state_outlines():
        """Lower-48 state borders as one None-separated (lon, lat) polyline."""
        states = json.loads((ASSETS / "us_states.geojson").read_text())
        lon, lat = [], []
        for feature in states["features"]:
            geom = feature["geometry"]
            polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
            for poly in polys:
                for ring in poly:
                    lon += [pt[0] for pt in ring] + [None]
                    lat += [pt[1] for pt in ring] + [None]
        return lon, lat

    STATE_OUTLINES = load_state_outlines()
    US = dict(lon_range=(-125, -66.5), lat_range=(24.3, 49.6))
    WEST = dict(lon_range=(-125, -102), lat_range=(31, 49.3))

    def base_map(lon_range, lat_range, height=480):
        """Pan/zoom map in lon/lat with state outlines; works offline."""
        lon, lat = STATE_OUTLINES
        fig = go.Figure(
            go.Scattergl(x=lon, y=lat, mode="lines", hoverinfo="skip",
                         line=dict(color="#B8B2A3", width=0.8), showlegend=False)
        )
        fig.update_xaxes(range=list(lon_range), visible=False)
        fig.update_yaxes(range=list(lat_range), visible=False,
                         scaleanchor="x", scaleratio=1.3)  # ~1/cos(40°)
        fig.update_layout(
            height=height, margin=dict(l=0, r=0, t=0, b=0), dragmode="pan",
            plot_bgcolor="#F7F5EF", paper_bgcolor="#F7F5EF",
            legend=dict(orientation="h", y=0, yanchor="top", itemsizing="constant"),
        )
        return fig

    def plant_trace(df, name, color):
        return go.Scattergl(
            x=df.lon, y=df.lat, mode="markers", name=name,
            marker=dict(size=3 + df.capacity_mw**0.5 / 3, color=color,
                        opacity=0.75, line=dict(width=0)),
            customdata=df[["plant_name", "capacity_mw", "ba", "fuel"]],
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:,.0f} MW · "
            "%{customdata[3]}<br>BA: %{customdata[2]}<extra></extra>",
        )

    def show_map(fig):
        return mo.ui.plotly(fig, config={"scrollZoom": True, "displaylogo": False})

    return US, WEST, base_map, plant_trace, plants, show_map


@app.cell(hide_code=True)
def _(terms, US, base_map, mo, plant_trace, plants, show_map):
    def fuel_map_slide():
        fuels = {
            "solar": "#E9C46A", "wind": "#2A9D8F", "hydro": "#2F6DB5", "gas": "#E07A1F",
            "coal": "#3D3D3D", "nuclear": "#8E44AD", "oil": "#8B5A2B",
            "waste": "#9AA0A6", "other": "#C9C2B0",
        }
        fig = base_map(**US)
        for fuel, color in fuels.items():
            df = plants[plants.fuel == fuel]
            fig.add_trace(plant_trace(df, f"{fuel} {df.capacity_mw.sum() / 1e3:,.0f} GW", color))

        return mo.vstack(
            [
                mo.md("## Every power plant in the lower 48"),
                show_map(fig),
                mo.md(
                    f"<small>{len(plants):,} operating plants, {plants.capacity_mw.sum() / 1e3:,.0f} GW, "
                    "sized by capacity. Source: EIA-860 (2025) via PUDL, Catalyst Cooperative. "
                    "Scroll to zoom, drag to pan, hover for details, click the legend to toggle fuels.</small>"
                ),
            ]
        )

    mo.vstack([fuel_map_slide(), terms(r"**capacity**: a plant’s maximum output (1 GW = 1,000 MW) &ensp;|&ensp; **BA** (balancing authority): keeps supply = demand in its area &ensp;|&ensp; **EIA-860**: federal survey of every power plant &ensp;|&ensp; **PUDL**: open-data project that cleans EIA data")])
    return


@app.cell(hide_code=True)
def _(terms, US, base_map, mo, plant_trace, plants, show_map):
    def iso_map_slide():
        isos = {  # EIA balancing-authority code: (market name, color)
            "CISO": ("CAISO", "#E73F74"), "ERCO": ("ERCOT", "#E68310"),
            "SWPP": ("SPP", "#80BA5A"), "MISO": ("MISO", "#3969AC"),
            "PJM": ("PJM", "#7F3C8D"), "NYIS": ("NYISO", "#11A579"),
            "ISNE": ("ISO-NE", "#F2B701"),
        }
        # SPP's RTO expanded into the Western Interconnection on 1 Apr 2026, after the
        # 2025 EIA-860 data: its West balancing area is mostly WAPA Rocky Mountain (WACM)
        # plus parts of WAPA Upper Great Plains West (WAUW).
        spp_west = plants.ba.isin(["WACM", "WAUW"])
        in_iso = plants.ba.isin(list(isos)) | spp_west
        share = plants[in_iso].capacity_mw.sum() / plants.capacity_mw.sum()

        fig = base_map(**US, height=420)
        fig.add_trace(plant_trace(plants[~in_iso], "Utility BAs, no ISO", "#B8B2A3"))
        for code, (name, color) in isos.items():
            fig.add_trace(plant_trace(plants[plants.ba == code], name, color))
        fig.add_trace(plant_trace(plants[spp_west], "SPP West (since Apr 2026)", "#C2DFA6"))

        return mo.vstack(
            [
                mo.md("## Where the markets are: ISOs and RTOs"),
                show_map(fig),
                mo.md(
                    f"**{share:.0%}** of US capacity is inside an ISO/RTO; the Southeast and most of the West are "
                    "vertically integrated utilities. In April 2026 **SPP expanded into the West**, the first RTO "
                    "in both interconnections.\n\n"
                    "<small>EIA-860 2025 balancing authorities via PUDL. SPP West ≈ the WACM and WAUW areas "
                    "(only parts of WAUW joined).</small>"
                ),
            ]
        )

    mo.vstack([iso_map_slide(), terms(r"**vertically integrated utility**: one company owns plants and wires and runs them without a market &ensp;|&ensp; **interconnection**: a synchronized grid (Eastern, Western, Texas) &ensp;|&ensp; **WACM, WAUW**: federal (WAPA) balancing areas in the Rockies and northern Plains")])
    return


@app.cell(hide_code=True)
def _(terms, WEST, base_map, mo, plant_trace, plants, show_map):
    WECC_BAS = [
        "AVA", "AVRN", "AZPS", "BANC", "BPAT", "CHPD", "CISO", "DEAA", "DOPD", "EPE",
        "GCPD", "GRID", "GRIF", "GRIS", "GWA", "IID", "IPCO", "LDWP", "NEVP", "NWMT",
        "PACE", "PACW", "PGE", "PNM", "PSCO", "PSEI", "SCL", "SRP", "TEPC", "TIDC",
        "TPWR", "WACM", "WALC", "WAUW", "WWA",
    ]

    def wecc_map_slide():
        west = plants[plants.ba.isin(WECC_BAS)]
        top = west.groupby("ba").capacity_mw.sum().nlargest(12).index
        colors = [
            "#E73F74", "#3969AC", "#E68310", "#11A579", "#7F3C8D", "#F2B701",
            "#80BA5A", "#008695", "#CF1C90", "#4B4B8F", "#A5AA99", "#8B5A2B",
        ]

        fig = base_map(**WEST, height=540)
        fig.add_trace(plant_trace(west[~west.ba.isin(top)], "Other WECC BAs", "#B8B2A3"))
        for ba, color in zip(top, colors):
            fig.add_trace(plant_trace(west[west.ba == ba], ba, color))

        return mo.vstack(
            [
                mo.md("## The West: a patchwork of balancing authorities"),
                mo.hstack(
                    [
                        show_map(fig),
                        mo.md(
                            f"""
    **{west.ba.nunique()} balancing authorities** in the Western Interconnection (WECC),
    each balancing its own supply and demand.

    Two RTOs: **CAISO** and, since April 2026, **SPP** (most of WACM, part of WAUW).

    CAISO also runs markets for other BAs: the **Western Energy Imbalance Market**
    (5-minute dispatch, the same kind of LP) and, since May 2026, the **Extended Day-Ahead
    Market** (EDAM), starting with PacifiCorp.

    <small>Largest 12 BAs by capacity in color. Source: EIA-860 (2025) via PUDL.</small>
    """
                        ),
                    ],
                    widths=[2.2, 1],
                ),
            ]
        )

    mo.vstack([wecc_map_slide(), terms(r"**dispatch**: setting each running unit’s MW output &ensp;|&ensp; **imbalance**: actual supply or demand minus its schedule &ensp;|&ensp; **WECC**: the Western Interconnection’s reliability council &ensp;|&ensp; **day-ahead market**: clears tomorrow’s hourly schedule today")])
    return


@app.cell(hide_code=True)
def _(terms, ASSETS, WEST, base_map, go, json, mo, show_map):
    western = json.loads((ASSETS / "western_16h_model.json").read_text())

    def transmission_map_slide(model):
        bus = {b["bus_i"]: b for b in model["bus"].values()}
        classes = [  # (label, min kV, color, width), drawn low voltage first
            ("< 100 kV", 0, "#D5CFC2", 0.5),
            ("100–199 kV", 100, "#A9B4C8", 0.7),
            ("200–299 kV", 200, "#5B85C0", 1.1),
            ("300–449 kV", 300, "#2F4A7A", 1.8),
            ("≥ 450 kV", 450, "#E07A1F", 2.4),
        ]
        segments = {label: ([], []) for label, *_ in classes}
        n_lines = 0
        for branch in model["branch"].values():
            if branch["transformer"] or branch["br_status"] != 1:
                continue
            a, b = bus[branch["f_bus"]], bus[branch["t_bus"]]
            label = [c[0] for c in classes if a["base_kv"] >= c[1]][-1]
            segments[label][0].extend([a["lon"], b["lon"], None])
            segments[label][1].extend([a["lat"], b["lat"], None])
            n_lines += 1

        fig = base_map(**WEST, height=555)
        for label, _, color, width in classes:
            xs, ys = segments[label]
            fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", name=label,
                                       line=dict(color=color, width=width),
                                       hoverinfo="skip"))

        return mo.vstack(
            [
                mo.md("## The western transmission network"),
                mo.hstack(
                    [
                        show_map(fig),
                        mo.md(
                            f"""
    **{len(model['bus']):,} buses** and **{n_lines:,} lines** in an open model of the
    Western Interconnection.

    The orange **500 kV** backbone moves bulk power between regions. Lower-voltage
    networks serve load locally.

    Each line has a reactance $x_\\ell$ and a thermal limit $F_\\ell$: exactly the
    data the DC-OPF LP needs.

    <small>Microsoft Research GridSFM US Power Grid dataset (MIT license), reconstructed from
    OpenStreetMap, EIA and Census data. Approximate: straight segments between substations,
    estimated parameters.</small>
    """
                        ),
                    ],
                    widths=[2.2, 1],
                ),
            ]
        )

    mo.vstack([transmission_map_slide(western), terms(r"**bus**: a network node, a substation at one voltage, where lines, generators and loads connect &ensp;|&ensp; **kV**: kilovolts &ensp;|&ensp; **reactance** $x_\ell$: how strongly a line resists AC flow &ensp;|&ensp; **DC-OPF**: the dispatch LP plus network constraints (coming up)")])
    return (western,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## How the market runs

    | Step | When | What it decides | Math |
    |:---|:---|:---|:---|
    | **1. Offers** | Before the market | Generators submit price–quantity offers | — |
    | **2. Day-ahead market** | The day before, hourly | Which plants turn **on**, and roughly how much they make | **MILP**: unit commitment |
    | **3. Real-time market** | Every **5 minutes** | Output of plants already on | **LP**: economic dispatch |

    Today: the LP inside step 3 first, then the MILP inside step 2.

    <small>**Terms** — **unit commitment**: deciding which units are on &ensp;|&ensp; **economic dispatch**: setting the MW of units already on</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## How generators bid, get awarded, and get paid

    1. **Bid:** each generator offers a supply curve, a list of steps like *“200 MW at \$25/MWh.”*
    2. **Clear:** the ISO picks the cheapest set of offers that meets demand (and respects the grid).
    3. **Award:** each generator is told how many MW to produce: its **award**.
    4. **Settle:** every awarded MWh is paid the **clearing price**, *not its own bid*.

    | Unit | Offer (\$/MWh) | Awarded (MW) | Paid (\$/MWh) | Margin (\$/MWh) |
    |:---|:---|:---|:---|:---|
    | Wind | 5 | 200 | 35 | 30 |
    | Nuclear | 12 | 250 | 35 | 23 |
    | Coal | 25 | 200 | 35 | 10 |
    | **Gas CC** | **35** | **170 of 200** (marginal) | 35 | 0 |
    | Peaker | 80 | 0 | — | — |

    *Demand 820 MW.* Since it is paid the clearing price whatever it bids, a small generator’s best strategy is to
    **bid its true marginal cost**.

    <small>**Terms** — **marginal cost**: cost of one more MWh &ensp;|&ensp; **Gas CC**: efficient combined-cycle plant &ensp;|&ensp; **peaker**: runs only at peaks &ensp;|&ensp; **small**: can’t move the price</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    ## Economic dispatch: who produces how much?

    Every 5 minutes (real time), and for each hour of tomorrow (day ahead), the ISO answers one question:

    /// admonition | The economic dispatch problem
    Which generators should produce how much, so that **supply = demand** at the **lowest total cost**?
    ///

    | Inputs | Output |
    |:---|:---|
    | Each generator’s offers (*“up to X MW at \$Y/MWh”*) and limits | A MW schedule for every unit |
    | The demand forecast | One price that every unit is paid |

    Before markets, a utility ran the same calculation on its own plants’ costs; an ISO runs it on offers.

    **This section:** the intuition first (the *merit order*), then the LP, then why the market price is its **dual**.
    """
    )
    return


@app.cell(hide_code=True)
def _(demand_slider, go, mo, pd, show_chart):
    merit = pd.DataFrame(
        {
            "unit": ["Wind", "Nuclear", "Coal", "Gas CC", "Peaker"],
            "capacity": [200, 250, 200, 200, 150],
            "cost": [5, 12, 25, 35, 80],
        }
    )
    merit["end"] = merit["capacity"].cumsum()
    merit["start"] = merit["end"] - merit["capacity"]

    def merit_order_slide(demand):
        marginal = merit[merit["end"] >= demand].iloc[0]
        price = marginal["cost"]

        fig = go.Figure()
        for _, row in merit.iterrows():
            if row["unit"] == marginal["unit"]:
                color = "#E07A1F"
            elif row["end"] <= demand:
                color = "#2F4A7A"
            else:
                color = "#C9C2B0"
            fig.add_trace(go.Bar(
                x=[row["start"] + row["capacity"] / 2], y=[row["cost"]], width=[row["capacity"]],
                marker=dict(color=color, line=dict(color="white", width=1.5)), showlegend=False,
                text=[row["unit"]], textposition="outside", cliponaxis=False,
                hovertemplate=f"{row['unit']}: {row['capacity']} MW at ${row['cost']}/MWh<extra></extra>",
            ))
        fig.add_vline(demand, line=dict(color="#14213D", width=2))
        fig.add_hline(price, line=dict(color="#A84F0A", dash="dash", width=1.5))
        fig.add_annotation(x=10, y=price, text=f"<b>Price = ${price}/MWh</b>", showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(color="#A84F0A"))
        fig.update_xaxes(title="Cumulative capacity (MW)", range=[0, 1000])
        fig.update_yaxes(title="Offer price ($/MWh)", range=[0, 95])

        return mo.vstack(
            [
                mo.md("## The merit order: stack offers cheapest first"),
                demand_slider,
                show_chart(fig, 400, bargap=0),
                mo.md(
                    f"Demand **{demand} MW** is met by everything left of the line. "
                    f"The last unit needed, **{marginal['unit']}**, sets the price for everyone: **\\${price}/MWh**."
                ),
            ]
        )

    merit_order_slide(demand_slider.value)
    return (merit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Economic dispatch is an LP

    $$
    \begin{aligned}
    \min_{p}\quad & \sum_i c_i\, p_i && & \color{#6A7179}\text{cost of the accepted offers} \\
    \text{s.t.}\quad & \sum_i p_i = D && (\lambda) & \color{#6A7179}\text{supply = demand} \\
    & \underline{P}_i \le p_i \le \overline{P}_i && (\underline\mu_i,\ \overline\mu_i) & \color{#6A7179}\text{output range of each online dispatchable unit} \\
    & 0 \le p_j \le \overline{P}^{\,\text{avail}}_j && (\overline\mu_j) & \color{#6A7179}\text{wind, solar: at most what the weather provides}
    \end{aligned}
    $$

    - $p_i$: output of generator $i$ (MW), the decision &ensp;|&ensp; $c_i$: its offer (\$/MWh) &ensp;|&ensp; $D$: demand
    - **Dual variables** (in parentheses, all in \$/MWh): $\lambda$ prices the supply = demand constraint;
      $\overline\mu_i \ge 0$ and $\underline\mu_i \ge 0$ price unit $i$’s maximum and minimum. Each is zero
      unless its constraint binds.
    - **Dispatchable units** (gas, coal, hydro, batteries) that are online run between their minimum
      $\underline P_i$ and maximum $\overline P_i$. Which units are online was decided earlier, in unit commitment.
    - **Wind and solar are not dispatchable:** the weather sets their available output $\overline{P}^{\,\text{avail}}_j$
      each interval. The LP can only **curtail** them below it.

    Substituting $p_i = \underline P_i + \delta_i$ leaves a **continuous knapsack**, so the greedy merit order solves it exactly.

    <small>**Terms** — **online**: committed (on) this interval &ensp;|&ensp; **curtail**: use less wind or solar than the weather allows &ensp;|&ensp; $i$ indexes all units (so $\sum_i p_i$ includes wind and solar); $j$ marks the wind and solar units, whose upper limit is $\overline P^{\,\text{avail}}_j$ &ensp;|&ensp; $\delta_i \ge 0$: output above the minimum</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # The market price *is* the dual variable

    | LP idea | Market meaning |
    |:---|:---|
    | **Shadow price** $\lambda$ | Cost of serving one more MWh, which is the clearing price. Every generator is paid $\lambda$. |
    | **Complementary slackness** | A unit strictly between $\underline P_i$ and $\overline P_i$ has $c_i = \lambda$, so **the marginal unit sets the price**. At its maximum $c_i \le \lambda$; held at its minimum, $c_i \ge \lambda$. |
    | **Strong duality** | $\lambda D = \sum_i c_i p_i + \sum_i \overline\mu_i \overline P_i - \sum_i \underline\mu_i \underline P_i$: consumer payments = production cost + profit of the cheaper units − losses of units held at their minimum. |

    <small>As on the previous slide: $\lambda$: price, dual of supply = demand ($\sum_i p_i = D$) &ensp;|&ensp; $p_i$: output of unit $i$ &ensp;|&ensp; $c_i$: its offer &ensp;|&ensp; $D$: demand &ensp;|&ensp; $\underline P_i, \overline P_i$: its min and max &ensp;|&ensp; $\underline\mu_i, \overline\mu_i \ge 0$: duals of those limits</small>
    """
    ).center()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Why the dual is the right price

    **1. It is the marginal cost of demand.** By LP sensitivity,
    $\lambda = \dfrac{\partial\, \text{(optimal cost)}}{\partial D}$: what one more MWh costs the system.

    **2. It makes the dispatch self-enforcing.** The ISO solves centrally, but at price $\lambda$ its schedule is
    each generator’s own profit-maximizing choice, so no one wants to deviate. Relax the balance constraint with $\lambda$:

    $$
    L(p, \lambda) = \sum_i c_i p_i - \lambda\Big(\sum_i p_i - D\Big)
    = \lambda D \;-\; \sum_i \underbrace{(\lambda - c_i)\, p_i}_{\text{generator } i\text{'s profit}}
    $$

    The problem **splits into one problem per generator**: at price $\lambda$, maximize your own profit
    $(\lambda - c_i)\,p_i$ within your own limits.

    <small>$\lambda$: price, the dual of supply = demand ($\sum_i p_i = D$) &ensp;|&ensp; $p_i$: output of generator $i$ (MW) &ensp;|&ensp; $c_i$: its offer (\$/MWh) &ensp;|&ensp; $D$: demand &ensp;|&ensp; $L$: the Lagrangian</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Each generator’s best response is the LP’s dispatch

    KKT stationarity, $c_i = \lambda - \overline\mu_i + \underline\mu_i$, is exactly each generator’s
    profit-maximizing condition:

    - $\lambda$ = the price (\$/MWh), the dual of supply = demand.
    - $\overline\mu_i = \lambda - c_i > 0$ only for a unit **at its maximum**: its profit per MWh, and what one more MW of its capacity would be worth.
    - $\underline\mu_i = c_i - \lambda > 0$ only for a unit **held at its minimum**: its loss per MWh at that price.

    | At $\lambda = \$35$ | Offer | Profit-maximizing choice |
    |:---|:---|:---|
    | Wind, nuclear, coal | below \$35 | run at maximum |
    | Gas CC | exactly \$35 | indifferent, so it fills the last 170 MW |
    | Peaker | \$80 | stay at minimum (here 0) |

    So **paying everyone $\lambda$ makes self-interested generators choose the LP’s optimal dispatch**,
    and supply meets demand: a competitive equilibrium. That is why ISOs settle at the dual.

    <small>$c_i$: unit $i$’s offer (\$/MWh) &ensp;|&ensp; $\overline\mu_i, \underline\mu_i$: duals of its max $\overline P_i$ and min $\underline P_i$, as before</small>
    """
    )
    return


@app.cell
def _(terms, QUIET, merit, mo, pypsa):
    def single_bus(demand):
        """Economic dispatch: min Σ c_i p_i  s.t.  Σ p_i = D (λ),  0 ≤ p_i ≤ Pmax_i."""
        n = pypsa.Network()
        n.add("Bus", "grid")                      # one bus → one balance constraint, dual λ
        # merit: the 5 offers on the merit-order slide, columns unit, capacity (MW), cost ($/MWh)
        for _, g in merit.iterrows():             # one decision variable p_i per generator
            n.add("Generator", g["unit"], bus="grid",
                  p_nom=g["capacity"],            # upper bound Pmax_i (MW)
                  marginal_cost=g["cost"])        # objective coefficient c_i ($/MWh offer)
        n.add("Load", "demand", bus="grid", p_set=demand)   # right-hand side D (MW)
        n.optimize(solver_name="highs", solver_options=QUIET)  # build the LP, solve with HiGHS
        return n                                  # λ is in n.buses_t.marginal_price

    mo.vstack([mo.md("## Code: economic dispatch as a PyPSA LP (one bus)"), mo.show_code(), terms(r"**PyPSA**: Python power-system modeling library &ensp;|&ensp; **HiGHS**: open-source LP/MILP solver &ensp;|&ensp; `p_nom` = $\overline P_i$ &ensp;|&ensp; `QUIET`: solver option that hides HiGHS’ log")])
    return (single_bus,)


@app.cell(hide_code=True)
def _(demand_slider, mo, single_bus):
    def single_bus_slide(demand):
        n = single_bus(demand)
        price = n.buses_t.marginal_price.iloc[0, 0]
        units = n.generators.assign(MW=n.generators_t.p.iloc[0].clip(lower=0) + 0.0)  # + 0.0 turns −0 into 0
        dispatch = mo.md(
            "| Unit | Marginal cost $c_i$ | Max MW | Dispatch MW | $\\lambda - c_i$ |\n|:---|---:|---:|---:|---:|\n"
            + "\n".join(
                f"| **{name}** | \\${g.marginal_cost:.0f}/MWh | {g.p_nom:.0f} | {g.MW:.0f} | {price - g.marginal_cost:+.0f} |"
                for name, g in units.iterrows()
            )
        )
        return mo.vstack(
            [
                mo.md("## The LP’s dual reproduces the merit-order price"),
                demand_slider,
                mo.hstack(
                    [
                        dispatch,
                        mo.stat(
                            value=f"${price:.0f}/MWh",
                            label="Dual of the balance constraint",
                            caption="matches the merit-order price",
                        ),
                    ],
                    justify="space-around",
                    align="center",
                ),
                mo.md(r"<small>$\lambda - c_i > 0$: run at max (profitable) &ensp;|&ensp; $= 0$: the marginal unit, sets the price &ensp;|&ensp; "
                      r"$< 0$: stays off. Exactly the per-generator problems from the Lagrangian.</small>"),
            ]
        )

    single_bus_slide(demand_slider.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    ## Background: what is a network flow problem?

    A classic LP family. A directed graph: each node $n$ has a supply or demand $b_n$, each arc has a cost $c_{ij}$
    per unit shipped and a capacity $u_{ij}$. Ship at least cost:
    $$
    \min \sum_{(i,j)} c_{ij} f_{ij} \quad \text{s.t.} \quad \underbrace{\textstyle\sum_j f_{nj} - \sum_j f_{jn} = b_n}_{\text{flow out − flow in}}\ \ \forall n, \qquad 0 \le f_{ij} \le u_{ij}
    $$

    - **Examples:** factories to warehouses (the transportation problem), pipelines, traffic, data networks.
    - **Special structure:** the constraint matrix is the node–arc *incidence matrix*, which is totally unimodular, so
      integer supplies and capacities give integer optimal flows, and the **network simplex** method is very fast.
    - **Duals:** each node’s conservation row has a dual, a *node price* (potential). Shipping from $i$ to $j$
      is worth the price difference, and a full arc earns the gap.

    **The grid is a network** (buses = nodes, lines = arcs), and the node prices will be *locational marginal prices* (LMPs). **The catch:** you
    can’t pick the path. Physics splits the flow, which adds constraints.

    <small>**Terms** — $f_{ij}$: flow on arc $i \to j$ &ensp;|&ensp; $b_n > 0$: supply, $< 0$: demand &ensp;|&ensp; **totally unimodular**: every square submatrix has determinant 0, ±1 &ensp;|&ensp; **arc**: a directed edge</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Adding the network: DC optimal power flow

    $$
    \begin{aligned}
    \min\quad & \sum_i c_i\, p_i && & \color{#6A7179}\text{total generation cost} \\
    \text{s.t.}\quad & \textstyle\sum_{i \in G(n)} p_i - D_n = \sum_{\ell} A_{n\ell}\, f_\ell && (\lambda_n) & \color{#6A7179}\text{each bus: generation − load = net flow out} \\
    & f_\ell = \tfrac{1}{x_\ell}\big(\theta_{a(\ell)} - \theta_{b(\ell)}\big) && & \color{#6A7179}\text{physics: flow = angle difference / reactance} \\
    & -F_\ell \le f_\ell \le F_\ell && (\mu_\ell) & \color{#6A7179}\text{each line’s thermal limit} \\
    & \underline P_i \le p_i \le \overline P_i && & \color{#6A7179}\text{dispatchable output range} \\
    & 0 \le p_j \le \overline P^{\,\text{avail}}_j && & \color{#6A7179}\text{wind, solar: curtail only}
    \end{aligned}
    $$

    - **One balance constraint per bus**, so **one price per bus**: the *locational marginal price* (LMP).
    - $A$ is the network’s **incidence matrix**: $A_{n\ell} = +1$ if line $\ell$ starts at bus $n$, $-1$ if it ends
      there, $0$ otherwise. So $\sum_\ell A_{n\ell} f_\ell$ is the **net flow out of bus $n$**.
    - **Summed over buses**, each column of $A$ sums to 0 (each line leaves one bus and enters another), so total generation
      = total load automatically. One angle is fixed as the reference, $\theta_{\text{ref}} = 0$.
    - Flows follow linearized physics (reactances $x_\ell$, angles $\theta$), or $f = \text{PTDF}\cdot\text{injections}$. Real AC power
      flow is nonconvex; this “DC” approximation keeps it an LP.

    <small>**Terms** — $n$: bus &ensp;|&ensp; $\lambda_n$: LMP at bus $n$ &ensp;|&ensp; $G(n)$: generators at bus $n$ &ensp;|&ensp; $D_n$: load at bus $n$ &ensp;|&ensp; $f_\ell$: MW on line $\ell$, from bus $a(\ell)$ to $b(\ell)$ &ensp;|&ensp; $\theta_n$: voltage angle at bus $n$ &ensp;|&ensp; $F_\ell$: line limit &ensp;|&ensp; $\mu_\ell$: value of 1 MW more line capacity &ensp;|&ensp; **PTDF**: MW on a line per MW injected at a bus &ensp;|&ensp; **injection**: generation − load at a bus</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## DC-OPF is a min-cost network flow problem, plus physics

    | Network flow | Power grid |
    |:---|:---|
    | Nodes, arcs | Buses, transmission lines |
    | Sources, sinks | Generators (with costs $c_i$), loads $D_n$ |
    | Flow conservation at each node | Supply = demand at each bus: the LMP constraints $(\lambda_n)$ |
    | Arc capacities | Line limits $-F_\ell \le f_\ell \le F_\ell$ $(\mu_\ell)$ |

    On its own, that is a **min-cost transshipment problem** (a *transport model*): power could follow the
    cheapest path, as on controllable HVDC links.

    **Kirchhoff’s voltage law** adds one *side constraint* per loop (a row beyond conservation and capacity, so it’s no longer a pure network-flow LP), $\sum_{\ell \in \text{loop}} \pm\, x_\ell\, f_\ell = 0$
    (equivalently $f_\ell = (\theta_{a(\ell)} - \theta_{b(\ell)})/x_\ell$). Power then **splits across every parallel
    path in inverse proportion to reactance** and can’t be routed. Still an LP, but the constraint
    matrix loses total unimodularity. Next: on two buses (one path, no loop) KVL adds nothing, so both models agree; on a three-bus loop, KVL splits the flow and they don’t.

    <small>**Terms** — **HVDC**: direct-current line whose flow the operator sets &ensp;|&ensp; **transport model**: flow-only model without Kirchhoff’s law &ensp;|&ensp; **±**: + if the loop follows the line’s direction, − if not &ensp;|&ensp; as before: $f_\ell$: MW on line $\ell$ (from bus $a(\ell)$ to $b(\ell)$) &ensp;|&ensp; $x_\ell$: reactance &ensp;|&ensp; $\theta$: voltage angle &ensp;|&ensp; $\lambda_n$: LMP at bus $n$ &ensp;|&ensp; $\mu_\ell$: dual of the line limit</small>
    """
    )
    return


@app.cell
def _(QUIET, mo, pypsa):
    def two_bus(line_limit):
        """Min-cost network flow: flow conservation at each bus (dual = LMP) + an arc capacity."""
        n = pypsa.Network()
        n.add("Bus", ["bus 1", "bus 2"])          # two balance constraints: λ_1, λ_2
        n.add("Generator", "Gen A", bus="bus 1", p_nom=200, marginal_cost=20)  # cheap, far away
        n.add("Generator", "Gen B", bus="bus 2", p_nom=200, marginal_cost=50)  # pricey, local
        n.add("Load", "Load", bus="bus 2", p_set=150)   # all demand sits at bus 2
        n.add("Line", "line 1-2", bus0="bus 1", bus1="bus 2",
              x=0.1,                              # reactance (only matters with loops)
              s_nom=line_limit)                   # −F ≤ f ≤ F, dual μ: set by the slider
        n.optimize(solver_name="highs", solver_options=QUIET)
        return n                                  # LMPs: n.buses_t.marginal_price

    mo.vstack([mo.md("## Code: DC-OPF as a network flow LP (two buses, one line limit)"), mo.show_code()])
    return (two_bus,)


@app.cell(hide_code=True)
def _(terms, go, limit_slider, mo, show_chart, two_bus):
    def two_bus_slide(line_limit):
        n = two_bus(line_limit)
        output = n.generators_t.p.iloc[0]
        lmp = n.buses_t.marginal_price.iloc[0]
        flow = n.lines_t.p0.iloc[0, 0]
        line_dual = abs(float(n.model.constraints["Line-fix-s-upper"].dual.values.ravel()[0]))
        rent = (lmp["bus 2"] - lmp["bus 1"]) * flow
        binding = flow >= line_limit - 1e-6

        fig = go.Figure()
        color = "#E07A1F" if binding else "#8A9BB8"
        fig.add_annotation(x=3, y=0, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                           text="", showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=5,
                           arrowcolor=color, standoff=22, startstandoff=22)
        flow_label = f"{flow:.0f} MW flowing · limit {line_limit:.0f} MW" + ("  (at limit)" if binding else "")
        fig.add_annotation(x=1.5, y=0.22, text=f"<b>{flow_label}</b>" if binding else flow_label,
                           showarrow=False, font=dict(size=15, color="#14213D"))
        for x, label in [
            (0, f"Bus 1<br>Gen A ($20): {output['Gen A']:.0f} MW<br>LMP ${lmp['bus 1']:.0f}"),
            (3, f"Bus 2<br>Gen B ($50): {output['Gen B']:.0f} MW · Load 150 MW<br>LMP ${lmp['bus 2']:.0f}"),
        ]:
            fig.add_annotation(x=x, y=-0.25, text=label.replace("$", "&#36;"),  # no MathJax
                               showarrow=False, yanchor="top",
                               font=dict(size=15))
        fig.add_trace(go.Scatter(x=[0, 3], y=[0, 0], mode="markers", hoverinfo="skip",
                                 marker=dict(size=34, color="#14213D"), showlegend=False))
        fig.update_xaxes(range=[-1.2, 4.2], visible=False)
        fig.update_yaxes(range=[-0.95, 0.45], visible=False)

        return mo.vstack(
            [
                mo.md(
                    "## Worked example: two buses, one line\n\n"
                    "Gen A (\\$20, 200 MW) at bus 1 · Gen B (\\$50, 200 MW) and 150 MW of load at bus 2"
                ),
                limit_slider,
                show_chart(fig, 210),
                mo.hstack(
                    [
                        mo.stat(f"${lmp['bus 1']:.0f}", label="LMP, bus 1"),
                        mo.stat(f"${lmp['bus 2']:.0f}", label="LMP, bus 2"),
                        mo.stat(f"${line_dual:.0f}", label="Line shadow price"),
                        mo.stat(f"${rent:,.0f}", label="Congestion rent (to FTR holders)"),
                    ],
                    justify="space-around",
                ),
            ]
        )

    mo.vstack([two_bus_slide(limit_slider.value), terms(r"**congestion rent**: what loads pay minus what generators receive, because prices differ across a full line &ensp;|&ensp; **FTR**: financial right to that rent (next slide)")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Reading the duals (line limit = 100 MW)

    | Dual | Value | Meaning |
    |:---|:---|:---|
    | $\lambda_1$ | **\$20** | Gen A has spare capacity, so one more MWh at bus 1 costs \$20. |
    | $\lambda_2$ | **\$50** | The line is full, so one more MWh at bus 2 must come from Gen B. |
    | $\mu_\ell$ | **\$30** | Each extra MW of line capacity saves \$30, and $\lambda_2 - \lambda_1 = \mu_\ell$. |
    | Congestion rent | **\$3,000** | Loads pay $150 \times 50 = 7{,}500$; generators receive $4{,}500$. The ISO collects the difference, $30 \times 100$, and pays it out (below). |

    **LMP = energy price + congestion price** (+ losses, in real markets).

    **Who gets the \$3,000?** The ISO passes it on to holders of **financial transmission rights**
    (FTRs; *congestion revenue rights* in CAISO and ERCOT, *TCCs* in NYISO). An FTR for $X$ MW from bus 1 to bus 2
    pays $X(\lambda_2 - \lambda_1)$ every hour. Rights are allocated or auctioned, mostly to LSEs and other
    transmission customers, so an LSE at bus 2 can **hedge congestion**: here a 100 MW FTR pays it \$3,000.

    <small>**Terms** — $\lambda_1, \lambda_2$: LMPs at buses 1 and 2 &ensp;|&ensp; $\mu_\ell$: dual of the line’s limit &ensp;|&ensp; **LSE** (load-serving entity): utility or retailer that buys wholesale power for its customers &ensp;|&ensp; **hedge**: contract that offsets a price risk &ensp;|&ensp; **losses**: energy lost as heat on lines, the third part of an LMP</small>
    """
    )
    return


@app.cell
def _(QUIET, mo, pypsa):
    def three_bus(limit_13=150):
        """Network flow + one KVL row for the loop: PyPSA Lines obey Kirchhoff, so no routing."""
        n = pypsa.Network()
        n.add("Bus", ["bus 1", "bus 2", "bus 3"])
        n.add("Generator", "Gen 1", bus="bus 1", p_nom=500, marginal_cost=20)   # cheap
        n.add("Generator", "Gen 2", bus="bus 2", p_nom=500, marginal_cost=50)   # expensive
        n.add("Load", "Load", bus="bus 3", p_set=300)
        # identical reactances x: an injection splits 2/3 on the direct line, 1/3 around
        # the two-line detour. PyPSA adds one KVL constraint for the loop automatically.
        n.add("Line", "1-2", bus0="bus 1", bus1="bus 2", x=0.1, s_nom=1000)  # never binds
        n.add("Line", "2-3", bus0="bus 2", bus1="bus 3", x=0.1, s_nom=1000)  # never binds
        n.add("Line", "1-3", bus0="bus 1", bus1="bus 3", x=0.1, s_nom=limit_13)  # binds
        n.optimize(solver_name="highs", solver_options=QUIET)
        return n

    mo.vstack([mo.md("## Code: network flow plus Kirchhoff’s law (three-bus loop, one limited line)"), mo.show_code()])
    return (three_bus,)


@app.cell(hide_code=True)
def _(go, mo, show_chart, three_bus):
    def three_bus_slide():
        n = three_bus()
        lmp = n.buses_t.marginal_price.iloc[0]
        output = n.generators_t.p.iloc[0]
        flows = n.lines_t.p0.iloc[0]

        pos = {"bus 1": (0, 0), "bus 2": (2, 0), "bus 3": (1, 1.6)}
        center = (1, 0.53)
        fig = go.Figure()
        for name, line in n.lines.iterrows():
            start, end = pos[line.bus0], pos[line.bus1]
            flow = flows[name]
            if flow < 0:  # draw the arrow in the direction power actually flows
                start, end = end, start
            binding = abs(flow) >= line.s_nom - 1e-6
            color = "#E07A1F" if binding else "#8A9BB8"
            if abs(flow) < 1e-6:
                fig.add_trace(go.Scatter(x=[start[0], end[0]], y=[start[1], end[1]], mode="lines",
                                         line=dict(color=color, width=4, dash="dot"),
                                         hoverinfo="skip", showlegend=False))
            else:
                fig.add_annotation(x=end[0], y=end[1], ax=start[0], ay=start[1],
                                   xref="x", yref="y", axref="x", ayref="y", text="",
                                   showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=4,
                                   arrowcolor=color, standoff=18, startstandoff=18)
            # label on the outside of the triangle, off the line
            mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            dx, dy = mid[0] - center[0], mid[1] - center[1]
            norm = (dx**2 + dy**2) ** 0.5
            label = f"{abs(flow):.0f} MW" + (" (at limit)" if binding else "")
            if abs(flow) < 1e-6:
                label = "0 MW (see next slide)"
            fig.add_annotation(x=mid[0] + 0.45 * dx / norm, y=mid[1] + 0.45 * dy / norm,
                               text=f"<b>{label}</b>" if binding else label, showarrow=False,
                               font=dict(size=15))
        labels = {
            "bus 1": f"Bus 1<br>Gen 1 ($20): {output['Gen 1']:.0f} MW",
            "bus 2": f"Bus 2<br>Gen 2 ($50): {output['Gen 2']:.0f} MW",
            "bus 3": "Bus 3<br>Load 300 MW",
        }
        for bus, (x, y) in pos.items():
            fig.add_annotation(x=x, y=y - 0.32 if y == 0 else y + 0.18,
                               text=f"{labels[bus]}<br>LMP ${lmp[bus]:.0f}".replace("$", "&#36;"),
                               showarrow=False,
                               yanchor="top" if y == 0 else "bottom", font=dict(size=15))
        fig.add_trace(go.Scatter(x=[x for x, _ in pos.values()], y=[y for _, y in pos.values()],
                                 mode="markers", marker=dict(size=30, color="#14213D"),
                                 hoverinfo="skip", showlegend=False))
        fig.update_xaxes(range=[-0.9, 2.9], visible=False)
        fig.update_yaxes(range=[-0.9, 2.3], visible=False)

        return mo.vstack(
            [
                mo.md("## You can’t route power: a three-bus loop"),
                mo.hstack(
                    [
                        show_chart(fig, 420),
                        mo.md(
                            f"""
    Three identical lines. Cheap Gen 1 (\\$20) and expensive Gen 2 (\\$50) serve a
    300 MW load at bus 3. Only **line 1–3** has a limit: **150 MW**.

    As a pure **network flow**, Gen 1 would serve all 300 MW: 150 MW direct and
    150 MW around through bus 2 (cost \\$6,000, price \\$20 everywhere).
    Kirchhoff’s law forbids that routing, so the LP uses **150 MW from each**
    (cost \\$10,500).

    And the LMP at bus 3 is **\\${lmp['bus 3']:.0f}/MWh**: *higher than either generator’s offer*.
    """
                        ),
                    ],
                    widths=[1.2, 1],
                ),
            ]
        )

    three_bus_slide()
    return


@app.cell(hide_code=True)
def _(mo, pd, three_bus):
    def loop_flow_slide():
        n = three_bus()
        g1, g2 = n.generators_t.p.iloc[0][["Gen 1", "Gen 2"]]
        # identical lines: an injection reaching bus 3 sends 2/3 on its direct line
        # and 1/3 around the other two lines
        rows = {  # MW on each line from each generator's trip to the load (+ = arrow direction)
            "Line 1→3": (2 / 3 * g1, 1 / 3 * g2),
            "Line 2→3": (1 / 3 * g1, 2 / 3 * g2),
            "Line 1→2": (1 / 3 * g1, -1 / 3 * g2),
        }
        trips = mo.md(
            f"| Flow on | Gen 1’s {g1:.0f} MW, bus 1 → load at bus 3 | Gen 2’s {g2:.0f} MW, bus 2 → load at bus 3 "
            "| Total flow |\n|:---|:---|:---|:---|\n"
            + "\n".join(f"| **{line}** | {a:+.0f} MW | {b:+.0f} MW | **{a + b:.0f} MW** |"
                        for line, (a, b) in rows.items())
        )

        return mo.vstack(
            [
                mo.md(
                    r"""
    ## Following the flows: superposition

    Power takes **every path**, split by reactance. With identical lines, the direct line
    has half the reactance of the two-line detour, so it carries **2/3**; the detour carries **1/3**.
    Treat each generator’s power as its own trip to bus 3, then add the trips:
    """
                ),
                trips,
                mo.md(
                    r"""
    On line 1–2 the two trips go in **opposite directions** and cancel.

    **Why Gen 1 stops at 150 MW.** Line 1–3 carries $\tfrac23 p_1 + \tfrac13 p_2 \le 150$ with $p_1 + p_2 = 300$,
    so $p_1 \le 150$. Every MW from Gen 1 loads the limited line twice as hard as a MW from Gen 2.

    **Why bus 3 costs \$80.** Serve one more MWh at bus 3 *without* adding flow to line 1–3:
    $\Delta p_1 + \Delta p_2 = 1$ and $\tfrac23 \Delta p_1 + \tfrac13 \Delta p_2 = 0$, so
    $\Delta p_2 = +2$, $\Delta p_1 = -1$, and the cost is $2(50) - 1(20) = \$80$.

    <small>$p_1, p_2$: output of Gen 1 (\$20) and Gen 2 (\$50), MW &ensp;|&ensp; $\Delta p_1, \Delta p_2$: change in each to serve the extra MWh</small>
    """
                ),
            ]
        )

    loop_flow_slide()
    return


@app.cell
def _(QUIET, mo, pypsa):
    # Network from the PyPSA example "Negative LMPs from Line Congestion",
    # after Kyri Baker's 3bus_LMPs (costs in $ here; € in the original).
    def baker_three_bus(extra_load_bus1=0.0):
        """Same loop physics; now the congested line sits next to the cheapest generator."""
        n = pypsa.Network()
        n.add("Bus", ["Bus1", "Bus2", "Bus3"])
        n.add("Generator", "Gen1", bus="Bus1", p_nom=100, marginal_cost=10)    # cheapest
        n.add("Generator", "Gen2", bus="Bus2", p_nom=100, marginal_cost=20)
        n.add("Generator", "Gen3", bus="Bus3", p_nom=100, marginal_cost=100)   # most expensive
        n.add("Load", "Load3", bus="Bus3", p_set=100)          # all demand at Bus3
        if extra_load_bus1:                                    # the experiment: add load at Bus1
            n.add("Load", "Load1", bus="Bus1", p_set=extra_load_bus1)
        n.add("Line", "Line12", bus0="Bus1", bus1="Bus2", x=1, s_nom=100)
        n.add("Line", "Line23", bus0="Bus2", bus1="Bus3", x=1, s_nom=100)
        n.add("Line", "Line13", bus0="Bus1", bus1="Bus3", x=1, s_nom=10)   # the only tight line
        n.optimize(solver_name="highs", solver_options=QUIET)
        return n                                               # Bus1's LMP comes out negative

    mo.vstack([mo.md("## Code: network flow plus Kirchhoff’s law (PyPSA negative-price example)"), mo.show_code()])
    return (baker_three_bus,)


@app.cell(hide_code=True)
def _(terms, baker_three_bus, bus1_load_switch, go, mo, show_chart):
    def negative_price_slide(add_load):
        n = baker_three_bus(1.0 if add_load else 0.0)
        lmp = n.buses_t.marginal_price.iloc[0]
        output = n.generators_t.p.iloc[0]
        flows = n.lines_t.p0.iloc[0]
        loads = n.loads.groupby("bus").p_set.sum()

        def dollars(value, escape=False):
            sign = "−" if value < -0.5 else ""
            return f"{sign}{'\\$' if escape else '$'}{abs(value):,.0f}"

        pos = {"Bus1": (0, 1.6), "Bus2": (2, 1.6), "Bus3": (1, 0)}
        fig = go.Figure()
        for name, line in n.lines.iterrows():
            start, end = pos[line.bus0], pos[line.bus1]
            if flows[name] < 0:  # draw arrows in the direction power flows
                start, end = end, start
            binding = abs(flows[name]) >= line.s_nom - 1e-6
            fig.add_annotation(x=end[0], y=end[1], ax=start[0], ay=start[1],
                               xref="x", yref="y", axref="x", ayref="y", text="",
                               showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=4,
                               arrowcolor="#E07A1F" if binding else "#8A9BB8",
                               standoff=17, startstandoff=17)
            mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            offset = (0.0, 0.18) if mid[1] > 1.5 else ((-0.42, 0) if mid[0] < 1 else (0.42, 0))
            label = f"{abs(flows[name]):.0f} MW" + (" (limit)" if binding else "")
            fig.add_annotation(x=mid[0] + offset[0], y=mid[1] + offset[1],
                               text=f"<b>{label}</b>" if binding else label, showarrow=False)
        for k, (bus, (x, y)) in enumerate(pos.items(), start=1):
            gen = f"Gen{k}"
            label = f"{bus}: Gen{k} (${n.generators.marginal_cost[gen]:.0f}) {abs(output[gen]):.0f} MW"
            if bus in loads:
                label += f"<br>load {loads[bus]:.0f} MW"
            label += f"<br>LMP {dollars(lmp[bus])}"
            fig.add_annotation(x=x, y=y + 0.22 if y > 1 else y - 0.25,
                               text=label.replace("$", "&#36;"), showarrow=False,
                               yanchor="bottom" if y > 1 else "top")
            fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", hoverinfo="skip", showlegend=False,
                                     marker=dict(size=28, color="#E73F74" if lmp[bus] < 0 else "#14213D")))
        fig.update_xaxes(range=[-1.0, 3.0], visible=False)
        fig.update_yaxes(range=[-0.9, 2.5], visible=False)

        saved = " (was \\$7,600: the extra load **saved \\$60**)" if add_load else ""
        return mo.vstack(
            [
                mo.md("## A negative price: more load, lower cost"),
                mo.hstack(
                    [
                        mo.vstack([show_chart(fig, 380), bus1_load_switch]),
                        mo.md(
                            f"""
    Line 1–3 is limited to **10 MW**, and every MW sent to Bus3 loads it: 2/3 MW from Gen1, 1/3 MW
    from Gen2. Per MW of that scarce line, Gen2 saves \\$80/(1/3) = \\$240 over Gen3; Gen1 only
    \\$90/(2/3) = \\$135. So the **cheapest unit isn’t used at all**.

    The LMP at Bus1 is **{dollars(lmp['Bus1'], escape=True)}/MWh**. One more MWh of load at Bus1 *pulls flow back*
    against the congested line, letting the system shift **+2 MWh to Gen2, −1 MWh from Gen3**:
    $2(20) - 1(100) = -60$.

    Total cost: **\\${n.objective:,.0f}**{saved}. Flip the switch to re-solve.

    Real negative prices usually come from wind and solar with production tax credits or from
    inflexible units; this one comes purely from **Kirchhoff’s law** and a congested line.

    <small>Example from the PyPSA docs, [“Negative LMPs from Line Congestion”](https://docs.pypsa.org/latest/examples/negative-prices-kvl-baker/),
    reproducing [K. Baker, *3bus_LMPs*](https://github.com/kyribaker/3bus_LMPs); see
    [Baker & Gangammanavar (2024)](https://doi.org/10.48550/arXiv.2403.19032),
    “Locational marginal prices obey DC circuit laws.”</small>
    """
                        ),
                    ],
                    widths=[1.1, 1],
                ),
            ]
        )

    mo.vstack([negative_price_slide(bus1_load_switch.value), terms(r"**production tax credit**: federal \$/MWh subsidy, so wind runs even at negative prices &ensp;|&ensp; **inflexible units**: plants that can’t cheaply turn down or off")])
    return


@app.cell
def _(terms, mo, pd, pypsa):
    def western_network(model):
        """model: the map's grid JSON (per-unit): tables bus, branch, gen, load, dcline (HVDC), each {id: record}."""
        base = model["baseMVA"]  # 100 MVA: multiply per-unit values by this to get MW
        n = pypsa.Network()

        bus = pd.DataFrame(model["bus"].values())     # 5,076 buses → 5,076 LMPs
        n.add("Bus", bus.bus_i.astype(str).tolist(),
              x=bus.lon.values, y=bus.lat.values, v_nom=1.0)  # v_nom=1: x stays per-unit

        br = pd.DataFrame(model["branch"].values()).query("br_status == 1")  # 9,511 in-service lines
        n.add("Line", ("L" + br["index"].astype(str)).tolist(),
              bus0=br.f_bus.astype(str).values, bus1=br.t_bus.astype(str).values,
              x=br.br_x.values, s_nom=br.rate_a.values * base)  # thermal limit, MW

        gen = pd.DataFrame(model["gen"].values()).query("gen_status == 1 and pmax > 0")  # 746 units
        n.add("Generator", ("G" + gen["index"].astype(str)).tolist(),
              bus=gen.gen_bus.astype(str).values, p_nom=gen.pmax.values * base,
              carrier=gen.fuel_type.values,
              marginal_cost=[c[-2] / base if len(c) >= 2 else 0.0   # linear cost term, $/MWh
                             for c in gen.cost])                   # (none: wind/solar, $0)

        load = pd.DataFrame(model["load"].values()).query("status == 1")
        n.add("Load", ("D" + load["index"].astype(str)).tolist(),
              bus=load.load_bus.astype(str).values, p_set=load.pd.values * base)

        hvdc = pd.DataFrame(model["dcline"].values())  # HVDC: controllable, no physics
        n.add("Link", ("HVDC" + hvdc["index"].astype(str)).tolist(),
              bus0=hvdc.f_bus.astype(str).values, bus1=hvdc.t_bus.astype(str).values,
              p_nom=hvdc.pmaxf.values * base, p_min_pu=-1)
        return n

    mo.vstack([mo.md("## Code: the same DC-OPF (network flow + Kirchhoff) on a 5,000-bus western grid"), mo.show_code(), terms(r"**per-unit**: values divided by a 100 MVA base (1.0 = 100 MW) &ensp;|&ensp; `rate_a`: line rating &ensp;|&ensp; `p_min_pu=-1`: an HVDC link can flow both ways")])
    return (western_network,)


@app.cell
def _(mo, terms):
    def path_limits(model):
        """WECC paths (interfaces): net flow on a set of lines between two BAs, within ± a limit."""
        # model["interface"]: 17 paths, each a record with name, branch_ids, limit (per-unit)
        return [(path["name"],
                 ["L" + b for b in path["branch_ids"]],    # each line in its own from→to direction
                 path["limit"] * model["baseMVA"])         # MW
                for path in model["interface"].values()]

    def add_path_limits(paths):
        """PyPSA hook: after building the LP, add two custom constraint rows per path."""
        # paths: path_limits(model), a list of (name, line names, limit in MW)
        def extra_functionality(n, snapshots):
            flow = n.model["Line-s"]                  # the LP's line-flow variables f_ℓ
            for name, lines, limit in paths:
                net = sum(flow.loc[:, line] for line in lines if line in n.lines.index)  # Σ f_ℓ
                n.model.add_constraints(net <= limit, name=f"Path-{name}-fwd")    # duals = path price
                n.model.add_constraints(net >= -limit, name=f"Path-{name}-rev")
        return extra_functionality

    mo.vstack([mo.md("## Code: adding WECC path limits as custom LP constraints"), mo.show_code(), terms(r"**WECC path (interface)**: lines between two areas with a combined flow limit &ensp;|&ensp; **path price**: its dual, \$/MWh of transfer &ensp;|&ensp; `extra_functionality`: PyPSA hook to add your own LP rows")])
    return add_path_limits, path_limits


@app.cell(hide_code=True)
def _(terms, 
    ASSETS,
    QUIET,
    WEST,
    add_path_limits,
    base_map,
    go,
    mo,
    pd,
    path_limits,
    show_map,
    western,
    western_network,
):
    def solve_or_load_western_opf():
        """Solve once (~40 s) and store in assets/; afterwards just read the files."""
        bus_file = ASSETS / "western_opf_buses.parquet"
        line_file = ASSETS / "western_opf_lines.parquet"
        path_file = ASSETS / "western_opf_paths.parquet"
        if bus_file.exists() and line_file.exists() and path_file.exists():
            return pd.read_parquet(bus_file), pd.read_parquet(line_file), pd.read_parquet(path_file)

        paths = path_limits(western)
        n = western_network(western)
        n.optimize(solver_name="highs", solver_options=QUIET,
                   extra_functionality=add_path_limits(paths))
        info = pd.DataFrame(western["bus"].values())
        info.index = info.bus_i.astype(str)
        buses = pd.DataFrame(
            {
                "name": info["name"],
                "kv": info["base_kv"],
                "lon": n.buses.x,
                "lat": n.buses.y,
                "lmp": n.buses_t.marginal_price.iloc[0],
            }
        )
        lines = n.lines[["bus0", "bus1", "s_nom"]].assign(flow=n.lines_t.p0.iloc[0])
        lines["binding"] = lines.flow.abs() >= lines.s_nom - 1e-3
        path_rows = []
        for name, path_lines, limit in paths:
            dual = sum(abs(float(n.model.constraints[f"Path-{name}-{side}"].dual.values.ravel()[0]))
                       for side in ("fwd", "rev"))
            members = [line for line in path_lines if line in n.lines.index]
            path_rows.append({"path": name.replace("_to_", " → "), "limit_mw": limit,
                              "price": dual, "lines": ",".join(members)})
        paths_df = pd.DataFrame(path_rows)
        buses.to_parquet(bus_file)
        lines.to_parquet(line_file)
        paths_df.to_parquet(path_file)
        return buses, lines, paths_df

    opf_buses, opf_lines, opf_paths = solve_or_load_western_opf()

    def western_lmp_slide(buses, lines, paths):
        def segments(subset):
            xs, ys = [], []
            for b0, b1 in zip(subset.bus0, subset.bus1):
                xs += [buses.lon[b0], buses.lon[b1], None]
                ys += [buses.lat[b0], buses.lat[b1], None]
            return xs, ys

        binding_paths = paths[paths.price > 1e-6].sort_values("price", ascending=False)
        path_lines = lines.loc[[l for ls in binding_paths["lines"] for l in ls.split(",")]]

        fig = base_map(**WEST, height=480)
        xs, ys = segments(lines[~lines.binding])
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", name="Lines",
                                   line=dict(color="#D5CFC2", width=0.6), hoverinfo="skip"))
        fig.add_trace(go.Scattergl(
            x=buses.lon, y=buses.lat, mode="markers", name="Buses (LMP)",
            marker=dict(size=5, color=buses.lmp, colorscale="Viridis",
                        cmin=buses.lmp.quantile(0.02), cmax=buses.lmp.quantile(0.99),
                        colorbar=dict(title="$/MWh", x=1.0)),
            customdata=buses[["name", "kv", "lmp"]],
            hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]:.0f} kV"
            "<br>LMP $%{customdata[2]:.2f}/MWh<extra></extra>",
        ))
        xs, ys = segments(path_lines)
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", name="Binding path (interface)",
                                   line=dict(color="#E07A1F", width=4), hoverinfo="skip"))
        xs, ys = segments(lines[lines.binding])
        fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", name="Binding line limit",
                                   line=dict(color="#E73F74", width=4), hoverinfo="skip"))

        northwest = buses[(buses.lat > 43) & (buses.lon < -116)].lmp.mean()
        southwest = buses[buses.lat < 36.5].lmp.mean()
        path_list = "; ".join(f"{row.path} (\\${row.price:.0f})" for row in binding_paths.itertuples())
        return mo.vstack(
            [
                mo.md("## Western LMPs from our PyPSA DC-OPF (simulated, not real prices)"),
                mo.hstack(
                    [
                        show_map(fig),
                        mo.vstack(
                            [
                                mo.stat(f"{len(binding_paths)} paths, {int(lines.binding.sum())} lines",
                                        label="Binding limits"),
                                mo.stat(f"${buses.lmp.quantile(0.05):.0f} to ${buses.lmp.quantile(0.95):.0f}",
                                        label="LMPs, 5th–95th percentile"),
                                mo.md(
                                    f"**Path limits** between regions split the West into price zones: the "
                                    f"Northwest averages **\\${northwest:.0f}/MWh**, the desert Southwest "
                                    f"**\\${southwest:.0f}**. Cheap hydro and wind are trapped behind the "
                                    f"binding paths."
                                ),
                            ]
                        ),
                    ],
                    widths=[2.0, 1.25],
                ),
                mo.md(
                    f"<small>Path duals (\\$/MWh of transfer): {path_list}. GridSFM western model, 4 pm 15 Jul 2024, "
                    "11 BAs; estimated line data and generator costs, one hour, no unit commitment. Shows how congestion splits prices; "
                    "<b>not representative of actual western LMPs</b>.</small>"
                ),
            ]
        )

    mo.vstack([western_lmp_slide(opf_buses, opf_lines, opf_paths), terms(r"**price zone**: region with similar LMPs, separated by binding limits")])
    return opf_buses, opf_lines, opf_paths


@app.cell(hide_code=True)
def _(ASSETS, json, mo, terms):
    def ercot_map_slide():
        """The most congested ERCOT map captured by scripts/capture_ercot_map.py."""
        best = json.loads((ASSETS / "ercot_best.json").read_text())
        (low, low_price), (high, high_price) = best["low"], best["high"]
        hour = best["interval"].split("ending ")[1]
        return mo.vstack(
            [
                mo.md("## What ISOs publish: an LMP map (ERCOT)"),
                mo.hstack(
                    [
                        mo.image((ASSETS / "ercot_best_map.png").read_bytes(), width=520),
                        mo.image((ASSETS / "ercot_best_legend.png").read_bytes(), height=440),
                        mo.md(
                            f"""
    Every dot is a pricing node. Every 5 minutes ERCOT solves the dispatch LP
    and publishes the dual at each node, shaded on this map.

    **One color everywhere** means no binding line limits: a single price, like our one-bus LP.

    **Color gradients** mean congestion: here prices run from **\\${low_price:.0f}** ({low}) to
    **\\${high_price:.0f}/MWh** ({high}) in the same 15 minutes.
    """
                        ),
                    ],
                    widths=[3, 0.6, 2.4],
                    align="center",
                ),
                mo.md(
                    f"<small>ERCOT real-time LMP map, {best['interval'][:10]}, {hour[:2]}:{hour[2:]} (most congested interval captured) · "
                    "[live map](https://www.ercot.com/content/cdr/contours/rtmLmp.html), every 5 min · all ISOs: Appendix G</small>"
                ),
            ]
        )

    mo.vstack([ercot_map_slide(), terms(r"**SCED**: ERCOT’s 5-minute dispatch LP (security-constrained economic dispatch) &ensp;|&ensp; **pricing node**: a bus with a published LMP &ensp;|&ensp; **HB_ / LZ_**: hub / load zone")])
    return


@app.cell(hide_code=True)
def _(terms, go, mo, pd, show_chart):
    # ERCOT Real-Time Settlement Point Prices, interval ending 01:45, 10 Nov 2026
    # https://www.ercot.com/content/cdr/html/real_time_spp.html
    ercot = pd.Series(
        {
            "HB_PAN": 13.49, "LZ_RAYBN": 20.92, "LZ_NORTH": 22.59, "HB_NORTH": 22.98,
            "HB_HOUSTON": 34.07, "LZ_HOUSTON": 34.60, "HB_BUSAVG": 35.81,
            "LZ_SOUTH": 47.92, "HB_SOUTH": 47.96, "LZ_CPS": 51.40,
            "HB_WEST": 73.67, "LZ_WEST": 142.98,
        }
    )

    def ercot_spread_slide():
        colors = [
            "#E07A1F" if "WEST" in k else "#2F6DB5" if k == "HB_PAN" else "#8A9BB8"
            for k in ercot.index
        ]
        fig = go.Figure(go.Bar(
            x=ercot.values, y=ercot.index, orientation="h", marker_color=colors,
            text=[f"${price:.2f}" for price in ercot.values], textposition="outside",
            cliponaxis=False, hovertemplate="%{y}: $%{x:.2f}/MWh<extra></extra>",
        ))
        fig.update_xaxes(title="$/MWh", range=[0, 165])
        fig.update_yaxes(autorange="reversed")

        return mo.vstack(
            [
                mo.md("## Real data: prices differ by location (ERCOT, Texas)"),
                mo.hstack(
                    [
                        show_chart(fig, 420, bargap=0.25),
                        mo.md(
                            """
    In the same 15 minutes, the **Panhandle** cleared at **\\$13.49/MWh** and **far West Texas** at **\\$142.98**.

    That spread is the signature of binding transmission constraints between regions.

    At **4:00 pm** the same day, all 15 hubs and load zones ERCOT publishes sat between **\\$35.11 and \\$37.17**.
    Congestion comes and goes.
    """
                        ),
                    ],
                    widths=[1.4, 1],
                ),
                mo.md(
                    "<small>Source: ERCOT Real-Time Settlement Point Prices, interval ending 01:45, 10 Nov 2026. "
                    "12 of ERCOT’s 15 hubs and load zones shown; HB = hub, LZ = load zone. Settlement point prices "
                "are LMP-based and include ERCOT’s reserve-scarcity (ORDC) adders.</small>"
                ),
            ]
        )

    mo.vstack([ercot_spread_slide(), terms(r"**hub / load zone**: average price over a set of buses / over a region’s load &ensp;|&ensp; **settlement point price**: the price used for payment there, over 15 minutes &ensp;|&ensp; **ORDC** (operating reserve demand curve): price adder that rises as reserves shrink")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    ## Why reserves and capacity? Reliability needs MW, not just MWh

    The energy LP buys **MWh** for the next interval. Two reliability questions remain:

    | Horizon | Question | Market answer |
    |:---|:---|:---|
    | **Minutes** | If the largest plant trips, or wind drops, who covers it right now? | **Operating reserves**: MW held back, co-optimized with energy |
    | **Years** | Will enough plants exist for the hottest hour of a future year? | **Resource adequacy**: capacity markets or RA requirements |

    **Why energy prices alone may not get plants built (the *missing money* problem):**
    - A **peaker** may run only a few hundred hours a year, and must earn back its fixed costs in those hours.
    - **Price and offer caps** clip exactly those scarcity hours (FERC caps ISO/RTO offers at \$1,000/MWh, or verified cost up to \$2,000).
    - **Demand barely responds**, so in a true shortage operators shed load instead of raising the price. Reliability is
      shared by everyone, so no single buyer pays for enough of it.

    Two responses, both in this section: **pay separately for MW** (capacity markets), or let **scarcity prices** run very
    high (ERCOT, energy-only, capped at \$5,000/MWh since 2022).

    <small>**Terms** — **peaker**: plant that runs only at peak demand &ensp;|&ensp; **shed load**: controlled blackouts &ensp;|&ensp; **resource adequacy (RA)**: enough capacity to meet peak plus a margin</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Co-optimizing reserves: add a constraint, get a price

    $$
    \begin{aligned}
    \min\quad & \sum_i \big(c_i p_i + b_i r_i\big) && & \color{#6A7179}\text{energy cost + reserve offers} \\
    \text{s.t.}\quad & \sum_i p_i = D && (\lambda) & \color{#6A7179}\text{supply = demand} \\
    & \sum_i r_i \ge R^{\text{req}} && (\sigma) & \color{#6A7179}\text{hold at least }R^{\text{req}}\text{ MW in reserve} \\
    & p_i + r_i \le \overline P_i && (\overline\mu_i) & \color{#6A7179}\text{energy + reserve fit in each unit} \\
    & p_i \ge \underline P_i,\quad r_i \ge 0 && & \color{#6A7179}\text{online dispatchable units}
    \end{aligned}
    $$

    - $R^{\text{req}}$ grows with the **largest single outage** and with **forecast error** in net load. Here only
      dispatchable units hold reserve; wind and solar enter through $0 \le p_j \le \overline P^{\,\text{avail}}_j$.
    - **Reserve price = $\sigma$**, the dual of the requirement.
    - **Coupling:** a MW held in reserve can’t produce energy, so when reserves are scarce the energy price rises too.

    <small>**Terms** — $i$: all units (wind and solar serve load but hold no reserve here) &ensp;|&ensp; $p_i, c_i$: energy output and offer, $\lambda$: energy price, as before &ensp;|&ensp; $\sigma$: reserve price (\$/MW per hour) &ensp;|&ensp; $r_i$: MW of reserve from unit $i$ &ensp;|&ensp; $b_i$: its reserve offer (\$/MW per hour) &ensp;|&ensp; $R^{\text{req}}$: required reserve (MW) &ensp;|&ensp; **net load**: load minus wind and solar output &ensp;|&ensp; $\overline\mu_i$ here prices the shared limit $p_i + r_i \le \overline P_i$</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Getting paid: energy and reserves

    A unit providing reserves is paid for **two separate products**:

    | Product | Paid | When |
    |:---|:---|:---|
    | **Energy** | $\lambda \times p_i$ (\$/MWh × MWh) | for what it actually produces |
    | **Reserve** | $\sigma \times r_i$ (\$/MW-h × MW held) | for standing by, used or not |
    | **Deployed reserve** | real-time price × extra MWh | only if the ISO calls on it |

    **Example: a coal unit held at its minimum** $\underline P_i$ earns $\lambda$ per MWh; its headroom $\overline P_i - \underline P_i$ can be **spinning reserve**, earning $\sigma$ per MW.

    - **Opportunity cost:** for a unit providing both, KKT stationarity gives $\sigma - b_i = \lambda - c_i$. The reserve
      price pays for the energy margin the unit gives up by holding back, which is why the two prices move together.
    - **Below cost:** if $\lambda$ is less than the unit’s cost at its minimum ($\underline\mu_i > 0$), its market revenue
      falls short, and that shortfall is what **uplift** covers.

    <small>**Terms** — **spinning reserve**: spare capacity on a unit that is already running &ensp;|&ensp; **deployed**: reserve the ISO actually calls on &ensp;|&ensp; as before: $\lambda, \sigma$: energy, reserve prices &ensp;|&ensp; $p_i, r_i$: energy, reserve MW &ensp;|&ensp; $c_i, b_i$: their offers &ensp;|&ensp; $\underline P_i, \overline P_i$: min, max &ensp;|&ensp; $\underline\mu_i$: dual of the min &ensp;|&ensp; **uplift**: a make-whole payment from the ISO, outside the price, covering that shortfall (more under unit commitment)</small>
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
    | **Energy-only** | ERCOT | No capacity product. Scarcity pricing (an operating-reserve demand curve adder) raises energy prices when reserves run short. |

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
    | **Price** (\$/MW-day) | Net CONE | **Cost of new entry** for a reference new plant (a gas combustion turbine or combined cycle; PJM now uses a combined cycle), **net** of the energy and reserve revenue it expects to earn: the “missing money” a new plant needs. |

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

    ## Unit commitment: deciding who is on

    So far, *which* units are running was given and the LP only chose their MW. **Unit commitment** decides on/off
    for each unit and each hour of tomorrow, and on/off is a yes/no decision.

    **Mixed-integer linear programming (MILP):** an LP in which some variables must be integers (here 0 or 1).
    - The feasible set is no longer convex, so the simplex method alone can’t solve it.
    - **Branch and bound:** solve the LP *relaxation* (let $0 \le u \le 1$); if some $u$ is fractional, split into
      $u = 0$ and $u = 1$ subproblems; prune any branch whose LP bound is worse than the best solution found.
      **Cutting planes** tighten the relaxation along the way.
    - NP-hard in general, yet commercial solvers (Gurobi, CPLEX) solve ISO-scale day-ahead problems within the
      market’s time window.

    **This section:** why the day-ahead market needs binaries, a 24-hour UC in PyPSA, and the catch: **a MILP has no duals**, so where does the price come from?

    <small>**Terms** — **LP relaxation**: the MILP with integrality dropped; its optimum bounds the MILP’s &ensp;|&ensp; **NP-hard**: no known method that is fast on every instance</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(terms, make_subplots, mo, show_chart):
    def why_milp_slide(p_min=150, p_max=400, cost=30, no_load=2000, startup=5000):
        fig = make_subplots(1, 2, horizontal_spacing=0.1, subplot_titles=(
            "Feasible outputs are not convex", "Cost jumps when the unit turns on"))
        fig.update_annotations(font_size=14)

        # feasible output: {0} ∪ [Pmin, Pmax], versus its LP relaxation [0, Pmax]
        fig.add_shape(type="line", x0=0, x1=p_max, y0=0.35, y1=0.35, row=1, col=1,
                      line=dict(color="#C9C2B0", width=10))
        fig.add_annotation(x=p_max / 2, y=0.5, text="LP relaxation: [0, Pmax] allows 20 MW",
                           showarrow=False, font=dict(color="#6A6A6A"), row=1, col=1)
        fig.add_shape(type="line", x0=p_min, x1=p_max, y0=0, y1=0, row=1, col=1,
                      line=dict(color="#2F4A7A", width=10))
        fig.add_scatter(x=[0], y=[0], mode="markers", marker=dict(size=14, color="#2F4A7A"),
                        hoverinfo="skip", showlegend=False, row=1, col=1)
        fig.add_annotation(x=p_min / 2, y=-0.22, text="off: 0", showarrow=False, row=1, col=1)
        fig.add_annotation(x=(p_min + p_max) / 2, y=-0.22, text="on: [Pmin, Pmax]",
                           showarrow=False, row=1, col=1)
        fig.update_xaxes(title="Output (MW)", range=[-30, p_max + 30], row=1, col=1)
        fig.update_yaxes(range=[-0.4, 0.7], visible=False, row=1, col=1)

        # cost: 0 when off; jumps by startup + no-load cost when the unit turns on
        on = list(range(p_min, p_max + 1, 10))
        fig.add_scatter(x=[0], y=[0], mode="markers", marker=dict(size=12, color="#E07A1F"),
                        showlegend=False, hovertemplate="off: $0 per hour<extra></extra>", row=1, col=2)
        fig.add_scatter(x=on, y=[startup + no_load + cost * p for p in on], mode="lines",
                        line=dict(color="#E07A1F", width=3), showlegend=False,
                        hovertemplate="%{x} MW: $%{y:,.0f} per hour<extra></extra>", row=1, col=2)
        fig.add_annotation(x=p_min, y=startup + no_load + cost * p_min,
                           ax=10, ay=startup + no_load + cost * p_max * 0.9, axref="x2", ayref="y2",
                           text="startup + no-load cost", showarrow=True, arrowhead=2,
                           xanchor="left", row=1, col=2)
        fig.update_xaxes(title="Output (MW)", row=1, col=2)
        fig.update_yaxes(title="$ per hour", row=1, col=2)

        return mo.vstack(
            [
                mo.md(
                    r"""
    ## Why the day-ahead market is a MILP

    - **Timing:** large fuel-burning (thermal) plants take hours to start, and once on must stay on for hours
      (minimum up/down times). On/off decisions have to be made **a day ahead**.
    - **Nonconvexity:** a unit that is on can’t run below its minimum, so output lies in
      $\{0\} \cup [\underline P, \overline P]$. No set of linear constraints describes that, but a binary does:
      $\underline P\,u \le p \le \overline P\,u,\ u \in \{0,1\}$.
    - **Fixed costs:** startup and no-load costs are paid only when the unit turns on.
    """
                ),
                show_chart(fig, 290, margin=dict(l=10, r=10, t=30, b=10)),
                mo.md(
                    "**Real time is mostly an LP** because commitments are already fixed (ISOs still commit some fast-start "
                    "units in real time): each online unit’s range "
                    r"$[\underline P, \overline P]$ is convex again, and the 5-minute dispatch only moves output within it."
                ),
            ]
        )

    mo.vstack([why_milp_slide(), terms(r"$u$: 1 = on, 0 = off &ensp;|&ensp; $p$: output (MW) &ensp;|&ensp; $\underline P, \overline P$: min, max when on &ensp;|&ensp; **startup / no-load cost**: cost to start / to stay on, per hour &ensp;|&ensp; **fast-start**: starts within an hour")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## The unit commitment MILP: 24 hours at once

    $$
    \begin{aligned}
    \min\quad & \sum_t \sum_i \big(c_i\, p_{it} + c^{\text{SU}}_i\, v_{it}\big) && \color{#6A7179}\text{energy + startup costs, all hours} \\
    \text{s.t.}\quad & \textstyle\sum_i p_{it} = D_t && \color{#6A7179}\text{supply = demand every hour} \\
    & \underline P_i\, u_{it} \le p_{it} \le \overline P_i\, u_{it} && \color{#6A7179}\text{on: output in range; off: exactly 0} \\
    & 0 \le p_{jt} \le \overline P^{\,\text{avail}}_{jt} && \color{#6A7179}\text{wind, solar: forecast, curtail only} \\
    & v_{it} \ge u_{it} - u_{i,t-1} && \color{#6A7179}\text{off → on counts as a start} \\
    & \textstyle\sum_{\tau = t-UT_i+1}^{t} v_{i\tau} \le u_{it} && \color{#6A7179}\text{started recently ⇒ must still be on} \\
    & u_{it},\ v_{it} \in \{0, 1\} && \color{#6A7179}\text{on/off and start are yes/no decisions}
    \end{aligned}
    $$

    - $u_{it} = 1$ if unit $i$ is **on** in hour $t$ &ensp;|&ensp; $v_{it} = 1$ if it **starts up** &ensp;|&ensp; $c^{\text{SU}}_i$ = startup cost
    - Multi-period (24–36 hours), coupled through time by startups, min up/down times and ramp limits.

    <small>**Terms** — $i$: all units, $j$: wind and solar &ensp;|&ensp; $\overline P^{\,\text{avail}}_{jt}$: forecast wind/solar output &ensp;|&ensp; **curtail**: produce less than that, e.g., when prices go negative or a line is full &ensp;|&ensp; $t$: hour &ensp;|&ensp; $p_{it}$: output (MW) &ensp;|&ensp; $c_i$: offer (\$/MWh) &ensp;|&ensp; $D_t$: demand in hour $t$ &ensp;|&ensp; $UT_i$: unit $i$’s minimum up time (hours) &ensp;|&ensp; $\tau$: the recent hours &ensp;|&ensp; **ramp limit**: most MW a unit can change per hour &ensp;|&ensp; no-load cost and minimum down time omitted for brevity</small>
    """
    )
    return


@app.cell
def _(QUIET, mo, np, pypsa):
    hours = np.arange(24)                         # one day, hourly
    demand_profile = 650 + 250 * np.sin((hours - 9) / 24 * 2 * np.pi)

    units = {  #  name: (capacity MW, $/MWh, min output, startup $, min up h)
        "Coal":   (450, 20, 0.5, 8000, 8),
        "Gas CC": (350, 35, 0.4, 3000, 4),
        "Peaker": (250, 80, 0.3, 1500, 1),
    }

    def unit_commitment(fixed_status=None):
        """MILP if fixed_status is None; otherwise the LP with binaries fixed.
        fixed_status: the MILP's on/off u_it, a 24 × 3 DataFrame of 0/1 (hour × unit)."""
        n = pypsa.Network()
        n.set_snapshots(hours)                     # 24 periods t, solved together
        n.add("Bus", "grid")
        n.add("Load", "demand", bus="grid", p_set=demand_profile)   # D_t
        for name, (cap, cost, pmin, startup, min_up) in units.items():
            if fixed_status is None:               # the MILP
                n.add("Generator", name, bus="grid", p_nom=cap, marginal_cost=cost,
                      committable=True,            # adds binaries u_it (on/off) and v_it (start)
                      p_min_pu=pmin,               # Pmin·u_it ≤ p_it ≤ Pmax·u_it
                      start_up_cost=startup,       # objective term c^SU_i v_it
                      min_up_time=min_up)          # Σ recent starts ≤ u_it
            else:                                  # the pricing LP: u_it fixed at the MILP's values
                on = fixed_status[name].values
                n.add("Generator", name, bus="grid", p_nom=cap, marginal_cost=cost,
                      p_max_pu=on, p_min_pu=pmin * on)   # same bounds, u_it now a constant
        n.optimize(solver_name="highs", solver_options=QUIET)  # HiGHS runs branch-and-bound
        return n

    mo.vstack([mo.md("## Code: 24-hour unit commitment as a PyPSA MILP"), mo.show_code()])
    return demand_profile, hours, unit_commitment, units


@app.cell(hide_code=True)
def _(demand_profile, go, hours, make_subplots, mo, show_chart, unit_commitment):
    uc = unit_commitment()
    status = uc.generators_t.status.round().astype(int)

    def uc_chart_slide():
        fig = make_subplots(2, 1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.1,
                            subplot_titles=("", "On/off status u<sub>it</sub> (dark = on)"))
        fig.update_annotations(font_size=13)
        for name, color in zip(uc.generators_t.p.columns, ["#2F4A7A", "#E07A1F", "#8A9BB8"]):
            fig.add_scatter(x=hours, y=uc.generators_t.p[name], name=name, stackgroup="dispatch",
                            mode="none", line=dict(shape="hv"), fillcolor=color,
                            hovertemplate=f"{name}: %{{y:.0f}} MW<extra></extra>", row=1, col=1)
        fig.add_scatter(x=hours, y=demand_profile, name="Demand", mode="lines",
                        line=dict(shape="hv", color="#14213D", width=2),
                        hovertemplate="Demand: %{y:.0f} MW<extra></extra>", row=1, col=1)
        fig.add_heatmap(x=hours, y=list(status.columns), z=status.T.values, zmin=0, zmax=1.6,
                        colorscale="Greys", showscale=False, xgap=1, ygap=1,
                        hovertemplate="%{y}, hour %{x}: u = %{z}<extra></extra>", row=2, col=1)
        fig.update_yaxes(title="MW", row=1, col=1)
        fig.update_yaxes(autorange="reversed", row=2, col=1)
        fig.update_xaxes(title="Hour", row=2, col=1)

        return mo.vstack([mo.md("## Solving the MILP with PyPSA + HiGHS"), show_chart(
            fig, 470, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", x=0.01, y=1.0, yanchor="top"),
        )])

    uc_chart_slide()
    return status, uc


@app.cell(hide_code=True)
def _(terms, go, itertools, mo, np, show_chart, units):
    def one_hour_cost(demand):
        """Exact minimum cost of serving `demand` for one hour with the UC units:
        try every on/off pattern (the MILP's binaries), dispatch the rest by merit order."""
        best = np.inf
        for on in itertools.product([0, 1], repeat=len(units)):
            committed = [u for u, is_on in zip(units.values(), on) if is_on]
            low = sum(cap * pmin for cap, _, pmin, _, _ in committed)
            high = sum(cap for cap, *_ in committed)
            if not low <= demand <= high:
                continue  # this on/off pattern can't serve this demand
            cost = sum(startup + c * cap * pmin for cap, c, pmin, startup, _ in committed)
            remaining = demand - low
            for cap, c, pmin, _, _ in sorted(committed, key=lambda u: u[1]):
                step = min(remaining, cap * (1 - pmin))
                cost, remaining = cost + c * step, remaining - step
            best = min(best, cost)
        return best

    def lower_convex_hull(xs, ys):
        hull = []
        for point in zip(xs, ys):
            while len(hull) >= 2 and (
                (hull[-1][0] - hull[-2][0]) * (point[1] - hull[-2][1])
                - (hull[-1][1] - hull[-2][1]) * (point[0] - hull[-2][0]) <= 0
            ):
                hull.pop()
            hull.append(point)
        return np.array(hull)

    def why_no_duals_slide():
        demand = np.arange(0, 1051, 5)
        cost = np.array([one_hour_cost(d) for d in demand])
        feasible = np.isfinite(cost)
        hull = lower_convex_hull(demand[feasible], cost[feasible])

        fig = go.Figure()
        fig.add_scatter(x=demand, y=np.where(feasible, cost, np.nan) / 1e3, mode="lines",
                        line=dict(color="#2F4A7A", width=2.5), name="MILP optimal cost C(D)",
                        hovertemplate="D = %{x} MW: $%{y:.1f}k<extra></extra>")
        fig.add_scatter(x=hull[:, 0], y=hull[:, 1] / 1e3, mode="lines",
                        line=dict(color="#E07A1F", width=2, dash="dash"),
                        name="Convex hull = Lagrangian dual (what a tight relaxation sees)",
                        hovertemplate="D = %{x} MW: $%{y:.1f}k<extra></extra>")
        fig.update_xaxes(title="Demand D (MW), one hour")
        fig.update_yaxes(title="Cost ($ thousand)")

        return mo.vstack(
            [
                mo.md(
                    r"""
    ## Why a MILP has no duals

    In an LP, optimal cost $C(D)$ is **convex** and the price is its slope, $\lambda = C'(D)$: a line with that
    slope **supports** the curve at the optimum, so every generator is happy with its LP output (strong duality).
    """
                ),
                show_chart(fig, 250, legend=dict(x=0.01, y=1, font=dict(size=12))),
                mo.md(
                    r"""
    With on/off decisions the feasible set is a **union** of polyhedra, so $C(D)$ is **not convex**: it **jumps**
    at startups and is undefined for 0–75 MW. Where $C(D)$ sits above its convex hull, **no line supports it**:
    no single price makes the MILP solution every generator’s best response. The Lagrangian dual leaves a
    **duality gap**, and branch-and-bound reports no duals.

    **In short:** a market price needs **strong duality**. LPs always have it; MILPs only where $C(D)$ touches its
    hull. **Convex hull pricing** uses the hull’s slope, the price that minimizes the gap, and so the uplift.
    """
                ),
            ]
        )

    mo.vstack([why_no_duals_slide(), terms(r"$C(D)$: least cost of serving $D$ MW for one hour &ensp;|&ensp; **uplift**: side payment for costs the market price doesn’t cover (next slide)")])
    return


@app.cell(hide_code=True)
def _(go, hours, mo, pd, show_chart, status, uc, unit_commitment, units):
    fixed = unit_commitment(fixed_status=status)
    lp_prices = fixed.buses_t.marginal_price["grid"]

    def uplift_table():
        rows = []
        for name, (cap, cost, pmin, startup, min_up) in units.items():
            output = fixed.generators_t.p[name]
            starts = int(status[name].diff().clip(lower=0).sum())
            revenue = float((lp_prices * output).sum())
            costs = float(cost * output.sum() + startup * starts)
            rows.append(
                {
                    "unit": name,
                    "startups": starts,
                    "revenue $": round(revenue),
                    "cost $": round(costs),
                    "uplift needed $": max(0, round(costs - revenue)),
                }
            )
        return pd.DataFrame(rows).set_index("unit")

    uplift = uplift_table()

    def pricing_slide():
        milp_prices = uc.buses_t.marginal_price["grid"]

        fig = go.Figure(go.Scatter(
            x=hours, y=lp_prices, mode="lines", line=dict(shape="hv", color="#E07A1F", width=2.5),
            hovertemplate="Hour %{x}: $%{y:.2f}/MWh<extra></extra>",
        ))
        fig.update_xaxes(title="Hour")
        fig.update_yaxes(title="$/MWh")

        return mo.vstack(
            [
                mo.md(
                    "## The pricing problem in practice\n\n"
                    f"PyPSA reports **{milp_prices.abs().max():.0f}** for every MILP dual: there are none to report. "
                    "So ISOs (and this demo) **fix the binaries at their optimal values, re-solve the LP and publish its duals**: the standard *pricing run*. "
                    "With binaries fixed, every unit’s range is convex again, so the LP has true duals:"
                ),
                mo.hstack([show_chart(fig, 290, margin=dict(l=10, r=10, t=40, b=10), title=dict(
                    text="Prices from the LP with binaries fixed", font=dict(size=14), x=0.5)),
                    uplift], widths=[1.2, 1]),
                mo.callout(
                    mo.md(
                        "At those prices, a unit that was switched on may not recover its startup cost or its losses at minimum output, "
                        "so the ISO pays it a make-whole **uplift** outside the price. "
                        "Alternatives price off a convex relaxation instead: **convex hull pricing**, and "
                        "MISO’s **ELMP** (Extended LMP), which relaxes fast-start units’ on/off variables "
                        "from {0, 1} to [0, 1] so their startup and no-load costs can set the price; PJM, NYISO, ISO-NE and SPP use similar fast-start pricing."
                    ),
                    kind="warn",
                ),
            ]
        )

    pricing_slide()
    return fixed, lp_prices, uplift


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Why unit commitment is hard in practice

    - **Size:** thousands of units, network models with tens of thousands of buses, 24–36 hourly periods.
      That means millions of variables and constraints.
    - **Deadline:** the day-ahead market has to clear in a fixed window, so solvers run under time limits with an optimality gap.
    - **Formulation:** tight formulations matter. An LP relaxation close to the convex hull lets branch-and-bound finish in time.

    US ISOs replaced Lagrangian relaxation with commercial MILP solvers in the 2000s–2010s (PJM first, around 2005),
    and FERC has credited the switch with large production-cost savings.

    <small>**Terms** — **FERC**: federal regulator of wholesale markets &ensp;|&ensp; **production cost**: total fuel, startup and operating cost of serving load</small>
    """
    )
    return


@app.cell
def _(ASSETS, np, pd):
    pge = pd.read_parquet(ASSETS / "pge_hourly.parquet")  # built by scripts/build_data.py

    def summer_scenarios(n_scenarios=200, n_days=30, lse_share=0.05, seed=0):
        """Bootstrap 30-day months from historical Jul–Sep days.

        Each sampled day keeps its own price, load and solar together, so their
        real co-movement (hot day → high load → high price) carries into the scenarios.
        """
        summer = pge[pge.index.month.isin([7, 8, 9])].copy()
        # solar output per MW of fleet, normalized within each year as the fleet grows
        daily_peak = summer.caiso_solar_mw.groupby(summer.index.date).max()
        fleet = daily_peak.groupby(pd.to_datetime(daily_peak.index).year).quantile(0.95)
        solar_mw = summer.caiso_solar_mw.clip(lower=0)  # night values are slightly negative
        summer["solar_per_mw"] = solar_mw / fleet.reindex(summer.index.year).values

        days = [day for _, day in summer.groupby(summer.index.date) if len(day) == 24]
        price = np.stack([day.price.values for day in days])
        load = np.stack([day.pge_load_mw.values for day in days]) * lse_share
        solar = np.stack([day.solar_per_mw.values for day in days])

        pick = np.random.default_rng(seed).integers(len(days), size=(n_scenarios, n_days))
        hour_of_day = np.tile(np.arange(24), n_days)
        flatten = lambda daily: daily[pick].reshape(n_scenarios, -1)
        dates = [day.index[0].date() for day in days]
        return flatten(price), flatten(load), flatten(solar), hour_of_day, dates

    price_scenarios, load_scenarios, solar_scenarios, hour_of_day, history_days = (
        summer_scenarios()
    )
    return (
        history_days,
        hour_of_day,
        load_scenarios,
        price_scenarios,
        solar_scenarios,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

    ## Background: what is portfolio optimization?

    **Markowitz (1952):** choose how much of each asset to hold, trading **expected return** against **risk**.
    Assets that don’t move together offset each other: *diversification*.
    $$
    \min_w\ w^\top \Sigma\, w \quad \text{s.t.} \quad m^\top w \ge \text{target}, \quad \textstyle\sum_k w_k = 1
    $$
    Measuring risk by variance makes this a **quadratic program**.

    - **Which risk?** Variance penalizes good surprises as much as bad ones. **CVaR** (Rockafellar & Uryasev, 2000) is
      the average of the worst outcomes, and with scenarios it becomes an **LP**.
    - **For an LSE** (a buyer facing the prices above): the assets are **contracts** (forwards, PPAs), the “return” is a **cost**, and the risk is
      months of high prices it can’t pass on to customers.
    - **Stochastic programming:** decide *now* (MW of each contract) before the uncertainty is revealed, then score
      the decision across **scenarios** of the future. Optimizing expected cost and CVaR over scenarios is a
      **(risk-averse) stochastic program**; scenarios turn the expectation and the CVaR into sums.

    <small>**Terms** — $w_k$: share in asset $k$ &ensp;|&ensp; $m$: expected returns &ensp;|&ensp; $\Sigma$: covariance matrix &ensp;|&ensp; **CVaR**: conditional value at risk &ensp;|&ensp; **scenario**: one possible future (here a month of prices and load)</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(
    terms,
    go,
    history_days,
    hour_of_day,
    load_scenarios,
    make_subplots,
    mo,
    np,
    price_scenarios,
    show_chart,
):
    unhedged_cost = (load_scenarios * price_scenarios).sum(axis=1) / 1e6  # $M per month

    def cvar(costs, alpha=0.95):
        """Average of the worst (1 − alpha) share of outcomes."""
        return np.sort(costs)[int(alpha * len(costs)):].mean()

    def lse_risk_slide():
        fig = make_subplots(1, 2, horizontal_spacing=0.08, subplot_titles=(
            "PG&E DLAP day-ahead price, summer days", "Unhedged cost distribution"))
        fig.update_annotations(font_size=14)
        by_hour = [price_scenarios[:, hour_of_day == h].ravel() for h in range(24)]
        bands = np.array([np.percentile(p, [5, 25, 50, 75, 95]) for p in by_hour])
        hours = np.arange(24)
        for lo, hi, color, label in [(0, 4, "#A9B4C8", "5–95%"), (1, 3, "#5B85C0", "25–75%")]:
            fig.add_scatter(x=hours, y=bands[:, lo], mode="lines", line=dict(width=0),
                            showlegend=False, hoverinfo="skip", row=1, col=1)
            fig.add_scatter(x=hours, y=bands[:, hi], mode="lines", line=dict(width=0), fill="tonexty",
                            fillcolor=color, name=label, hoverinfo="skip", row=1, col=1)
        fig.add_scatter(x=hours, y=bands[:, 2], mode="lines", line=dict(color="#14213D", width=2),
                        name="Median", hovertemplate="Hour %{x}: median $%{y:.0f}/MWh<extra></extra>",
                        row=1, col=1)
        fig.update_xaxes(title="Hour of day", row=1, col=1)
        fig.update_yaxes(title="$/MWh", row=1, col=1)

        size = (unhedged_cost.max() - unhedged_cost.min()) / 30 * (1 + 1e-9)  # 30 bins, max included
        fig.add_histogram(x=unhedged_cost, xbins=dict(start=unhedged_cost.min(),
                          end=unhedged_cost.max() + size / 2, size=size),
                          marker_color="#8A9BB8", showlegend=False, legend="legend2",
                          hovertemplate="$%{x}M: %{y} months<extra></extra>", row=1, col=2)
        fig.add_vline(unhedged_cost.mean(), line=dict(color="#14213D", width=2), row=1, col=2,
                      showlegend=True, name="Expected", legend="legend2")
        fig.add_vline(cvar(unhedged_cost), line=dict(color="#E73F74", width=2, dash="dash"),
                      row=1, col=2, showlegend=True, name="CVaR 95%", legend="legend2")
        fig.update_xaxes(title="Monthly procurement cost ($M)", row=1, col=2)

        return mo.vstack(
            [
                mo.md(
                    "## An LSE’s risk: volatile prices, fixed retail rates\n\n"
                    "An LSE in CAISO buys its customers’ load at the **DLAP price** every hour but sells at fixed "
                    "retail rates, so a price spike is a loss it can’t pass on. With the forecast as **scenarios**, "
                    "the monthly bill becomes a distribution; **hedges** (fixed-price contracts) narrow it."
                ),
                show_chart(
                    fig, 250, margin=dict(l=10, r=10, t=30, b=10), bargap=0.05,
                    legend=dict(x=0.01, y=1, font=dict(size=12)),
                    legend2=dict(x=1, xanchor="right", y=1, font=dict(size=12),
                                 bgcolor="#F7F5EF", traceorder="normal"),
                ),
                mo.md(
                    f"Expected cost **\\${unhedged_cost.mean():.1f}M**; the worst 5% of months average "
                    f"**\\${cvar(unhedged_cost):.1f}M** (CVaR). Here the forecast resamples history; others use "
                    "fundamentals models (the dispatch LPs above) or forward curves.\n\n"
                    f"<small>200 thirty-day months resampled from {len(history_days)} Jul–Sep days, "
                    f"{history_days[0]:%Y}–{history_days[-1]:%Y}: CAISO day-ahead PG&E DLAP prices (OASIS) with "
                    "the same day’s PG&E-area load (EIA-930 via PUDL), scaled to an LSE serving 5%.</small>"
                ),
            ]
        )

    mo.vstack([lse_risk_slide(), terms(r"**DLAP** (default load aggregation point): CAISO’s utility-area average price that load pays &ensp;|&ensp; **retail rate**: fixed price charged to end customers &ensp;|&ensp; **fundamentals model**: forecast by simulating dispatch &ensp;|&ensp; **forward curve**: traded prices for future delivery")])
    return cvar, unhedged_cost


@app.cell(hide_code=True)
def _(
    hour_of_day,
    load_scenarios,
    make_subplots,
    mo,
    np,
    price_scenarios,
    show_chart,
    solar_scenarios,
    terms,
):
    def at_risk_slide():
        hours = np.arange(24)
        by_hour = lambda x: np.array([x[:, hour_of_day == h].ravel() for h in hours])
        load, price = by_hour(load_scenarios), by_hour(price_scenarios)
        solar = by_hour(solar_scenarios).mean(axis=1)
        on_peak = ((hours >= 6) & (hours < 22)) * 1.0

        fig = make_subplots(1, 2, horizontal_spacing=0.12,
                            specs=[[{"secondary_y": True}, {"secondary_y": True}]],
                            subplot_titles=("What the LSE must buy, hour by hour",
                                            "What 1 MW of each contract delivers"))
        fig.update_annotations(font_size=14)
        fig.add_scatter(x=hours, y=np.percentile(load, 5, axis=1), mode="lines", line=dict(width=0),
                        showlegend=False, hoverinfo="skip", row=1, col=1)
        fig.add_scatter(x=hours, y=np.percentile(load, 95, axis=1), mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor="#A9B4C8", name="LSE load, 5–95%", hoverinfo="skip",
                        row=1, col=1)
        fig.add_scatter(x=hours, y=load.mean(axis=1), mode="lines", line=dict(color="#14213D", width=2),
                        name="LSE load, average", hovertemplate="Hour %{x}: %{y:,.0f} MW<extra></extra>",
                        row=1, col=1)
        fig.update_yaxes(title="MW", row=1, col=1, secondary_y=False)

        for y, name, line in [
            (np.ones(24), "Flat forward", dict(shape="hvh", color="#2F4A7A", width=4)),
            (on_peak, "On-peak forward", dict(shape="hvh", color="#E07A1F", width=2, dash="dash")),
            (solar, "Solar PPA (average day)", dict(color="#2A9D8F", width=2.5)),
        ]:
            fig.add_scatter(x=hours, y=y, mode="lines", line=line, name=name, legend="legend2",
                            hovertemplate=f"{name}, hour %{{x}}: %{{y:.2f}}<extra></extra>", row=1, col=2)
        fig.update_yaxes(title="MW delivered per MW bought", range=[0, 1.25], row=1, col=2,
                         secondary_y=False)

        for col, legend in [(1, "legend"), (2, "legend2")]:
            fig.add_scatter(x=hours, y=price.mean(axis=1), mode="lines", name="Price, average",
                            line=dict(color="#E73F74", width=2, dash="dash"), legend=legend,
                            hovertemplate="Hour %{x}: $%{y:.0f}/MWh<extra></extra>",
                            row=1, col=col, secondary_y=True)
            fig.update_yaxes(title=dict(text="$/MWh", font=dict(color="#E73F74")),
                             tickfont=dict(color="#E73F74"), row=1, col=col, secondary_y=True)
            fig.update_xaxes(title="Hour of day", tickvals=[0, 6, 12, 18, 23], row=1, col=col)

        daily_load = load_scenarios.reshape(len(load_scenarios), -1, 24).mean(axis=2).ravel()
        daily_price = price_scenarios.reshape(len(price_scenarios), -1, 24).mean(axis=2).ravel()
        corr = np.corrcoef(daily_load, daily_price)[0, 1]
        evening = (hours >= 17) & (hours < 21)

        return mo.vstack(
            [
                mo.md("## What is at risk: load, prices and contract shapes"),
                show_chart(
                    fig, 350, margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(orientation="h", x=0.22, xanchor="center", y=-0.25, yanchor="top",
                                font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
                    legend2=dict(orientation="h", x=0.78, xanchor="center", y=-0.25, yanchor="top",
                                 font=dict(size=12), traceorder="normal"),
                ),
                mo.md(
                    f"""
    **The risk:** load peaks in the evening, exactly when prices peak (5–9 pm averages
    **{load.mean(axis=1)[evening].mean():,.0f} MW at \\${price.mean(axis=1)[evening].mean():.0f}/MWh**), and hot days raise both
    together (daily correlation **{corr:.2f}**). Every MWh not covered by a contract is bought at that price.

    **The tools:** a flat forward fixes the price on the same MW every hour; an on-peak block covers 6:00–22:00;
    a solar PPA delivers at midday, when power is cheapest, so it hedges the least.
    """
                ),
                terms(r"**exposed energy / open position**: load not covered by contracts, bought at the market price"),
            ]
        )

    at_risk_slide()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Setting up the hedging problem

    **Data.** $S = 200$ equally likely monthly scenarios $s$, $T = 720$ hours $t$: price $\lambda_{st}$ (\$/MWh), load $L_{st}$ (MW).

    **Hedges** are **fixed-price contracts**: the LSE buys $q_k$ MW in delivery shape $h_{kt}$ at a fixed
    price $K_k$ and buys the rest of its load at the market price $\lambda_{st}$; in hours when contracts exceed
    load, the **excess energy is sold at the market price**.

    | Contract $k$ | Delivery shape $h_{kt}$ (per MW) | Fixed price $K_k$ | Max MW offered $\bar q_k$ |
    |:---|:---|:---|:---|
    | **Flat forward** | 1 every hour | forecast average + \$1.00 | 400 MW |
    | **On-peak forward** | 1 from 6:00 to 22:00 (western block, simplified) | forecast on-peak + \$1.50 | 600 MW |
    | **Solar PPA** | solar fleet output that day (varies by scenario) | forecast solar-weighted + \$1.50 | 600 MW |

    **Decision** $q_k \ge 0$: MW of contract $k$. **Cost in scenario $s$**, linear in $q$:
    $$
    C_s(q) = \underbrace{\textstyle\sum_t \big(L_{st} - \sum_k q_k h_{kt}\big) \lambda_{st}}_{\text{buy the rest (or sell the excess) at market}}
    \;+\; \underbrace{\textstyle\sum_k q_k \sum_t h_{kt}\, K_k}_{\text{contracted energy at fixed prices}}
    $$


    <small>**Terms** — **forward**: fixed MW at a fixed price, delivered later &ensp;|&ensp; **PPA**: power purchase agreement, a plant’s output at a fixed price &ensp;|&ensp; **premium** (+\$1.00 / +\$1.50): seller’s charge above forecast value for taking price risk (without it the LP hedges to the max) &ensp;|&ensp; **forecast** average / on-peak / solar-weighted: mean price over delivered hours</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(cvar, go, mo, np, show_chart, unhedged_cost):
    def risk_measures_slide(alpha=0.95):
        var = np.quantile(unhedged_cost, alpha)
        tail = np.sort(unhedged_cost)[int(alpha * len(unhedged_cost)):]

        fig = go.Figure()
        size = (unhedged_cost.max() - unhedged_cost.min()) / 34 * (1 + 1e-9)  # 34 bins, max included
        bins = dict(start=unhedged_cost.min(), end=unhedged_cost.max() + size / 2, size=size)
        fig.add_histogram(x=unhedged_cost, xbins=bins, marker_color="#C9C2B0", name="All 200 months",
                          hovertemplate="$%{x}M: %{y} months<extra></extra>")
        fig.add_histogram(x=tail, xbins=bins, marker_color="#E73F74",
                          name=f"Worst {1 - alpha:.0%} ({len(tail)} months)",
                          hovertemplate="$%{x}M: %{y} months<extra></extra>")
        for x, color, dash, label in [
            (unhedged_cost.mean(), "#14213D", "solid", f"Expected ${unhedged_cost.mean():.1f}M"),
            (var, "#E07A1F", "dot", f"VaR ${var:.1f}M"),
            (cvar(unhedged_cost), "#E73F74", "dash", f"CVaR ${cvar(unhedged_cost):.1f}M"),
        ]:
            fig.add_vline(x, line=dict(color=color, width=2, dash=dash), showlegend=True, name=label)
        fig.update_xaxes(title="Monthly procurement cost, unhedged ($M)")

        return mo.vstack(
            [
                mo.md("## Measuring risk: VaR and CVaR"),
                mo.hstack(
                    [
                        show_chart(fig, 360, barmode="overlay", bargap=0.05,
                                   legend=dict(x=1, xanchor="right", y=1, font=dict(size=12))),
                        mo.md(
                            rf"""
    Pick a confidence level, here $\alpha = 0.95$, and look at the **worst 5%** of months
    (10 of 200 scenarios).

    - **VaR** (value-at-risk): the cost exceeded in only 5% of months, the 95th percentile.
    - **CVaR** (conditional value-at-risk): the **average cost in those worst 5%**. It answers
      *“when a bad month happens, how bad is it on average?”* Unlike VaR it accounts for how
      far into the tail the bad months go, and it is convex and, as the next slide shows, LP-representable.
    - $\bar R$, the **risk limit**: the LSE’s **risk tolerance**, the highest CVaR (\$M per month)
      it will accept, set by management or a regulator. It is the slider on the next slides.
    """
                        ),
                    ],
                    widths=[1.15, 1],
                ),
            ]
        )

    risk_measures_slide()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Choosing hedges is an LP: a CVaR limit

    **Goal:** minimize expected cost, keeping CVaR $\le$ the risk limit $\bar R$: a *two-stage stochastic program* (below).

    **Key identity** (Rockafellar–Uryasev): CVaR is itself a minimization over a cost threshold $z$,
    $$
    \text{CVaR}_\alpha(C) = \min_z \Big\{\, z + \tfrac{1}{(1-\alpha)S} \textstyle\sum_s \max\big(C_s - z,\ 0\big) \Big\},
    $$
    where $z$ is a free variable, a guess at the VaR, and the sum is the total excess over $z$ divided by the tail size $(1-\alpha)S = 10$. If $z$ = VaR, exactly the worst 10 months exceed it, so $z$ + their average excess = their average cost = CVaR; no other $z$ gives less (next slide), so the LP picks $z$ itself. $\max(\cdot, 0)$ isn’t linear, so give each scenario a variable $\eta_s$ with
    $\eta_s \ge C_s - z$ and $\eta_s \ge 0$: at the optimum, $\eta_s$ is how far scenario $s$’s cost exceeds $z$.

    $$
    \begin{aligned}
    \min_{q,\, z,\, \eta}\quad & \tfrac{1}{S} \textstyle\sum_s C_s(q) && \text{expected monthly cost} \\
    \text{s.t.}\quad & z + \tfrac{1}{(1-\alpha)S} \textstyle\sum_s \eta_s \le \bar R && \text{CVaR} \le \text{risk tolerance} \\
    & \eta_s \ge C_s(q) - z \quad \forall s && \text{scenario } s\text{'s cost above } z \\
    & \eta_s \ge 0 \quad \forall s && \text{costs below } z \text{ don't count} \\
    & 0 \le q_k \le \bar q_k \quad \forall k && \text{MW available in each contract}
    \end{aligned}
    $$

    <small>$q_k$: MW of contract $k$ &ensp;|&ensp; $C_s(q)$: cost in scenario $s$ &ensp;|&ensp;
    $z$: cost threshold, chosen by the LP (the VaR at the optimum) &ensp;|&ensp; $\eta_s$: excess over $z$ &ensp;|&ensp; $\alpha = 0.95$ &ensp;|&ensp; $\bar R$: risk limit (\$M) &ensp;|&ensp;
    $C_s(q)$ is linear in $q$, so this is an **LP**:
    204 variables, 201 constraints. **Two stages:** pick $q$ now; in each scenario, buy or sell the rest at market.</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Why minimizing over $z$ gives the CVaR: a small example

    Take $S = 20$ scenarios and $\alpha = 0.9$, so the tail is the worst $(1-\alpha)S = 2$. The three highest
    costs are **25, 30 and 40** (\$M); CVaR = average of the worst two = **35**.

    | Guess $z$ | Scenarios above $z$ | Total excess $\sum_s \max(C_s - z, 0)$ | $\div\,(1-\alpha)S = 2$ | $z$ + that |
    |:---|:---|:---|:---|:---|
    | 20 | 25, 30, 40 | 5 + 10 + 20 = 35 | 17.5 | 37.5 |
    | 25 | 30, 40 | 5 + 15 = 20 | 10 | **35** |
    | 30 | 40 | 10 | 5 | **35** |
    | 35 | 40 | 5 | 2.5 | 37.5 |

    - **$z$ too low:** scenarios outside the tail count as excess, so the sum is too big.
    - **$z$ too high:** $z$ itself overshoots, and the excess no longer makes up the difference.
    - **$z$ at the tail’s edge** (anywhere from 25 to 30, the VaR): the minimum, **35 = CVaR**. So the VaR is *a* best
      $z$, though not always the only one.

    In the LP, $\eta_s$ holds each $\max(C_s - z, 0)$, so the CVaR row is this same expression, written linearly.
    """
    )
    return


@app.cell
def _(terms, hour_of_day, load_scenarios, mo, np, price_scenarios, solar_scenarios):
    def fair_price(shape):
        """Forecast value of a delivery shape: price averaged over its MWh, in $/MWh."""
        weights = np.broadcast_to(shape, price_scenarios.shape)
        return (price_scenarios * weights).sum() / weights.sum()

    flat = np.ones(hour_of_day.shape)
    on_peak = ((hour_of_day >= 6) & (hour_of_day < 22)) * 1.0  # western HE7–HE22
    shapes = {"Flat forward": flat, "On-peak forward": on_peak, "Solar PPA": solar_scenarios}
    fair_values = np.array([fair_price(shape) for shape in shapes.values()])
    default_prices = fair_values + np.array([1.0, 1.5, 1.5])      # fixed price = forecast value + premium
    default_caps = [400, 600, 600]                                # MW
    hedges = {name: (shape, price, cap) for (name, shape), price, cap
              in zip(shapes.items(), default_prices, default_caps)}

    # Everything the LP needs is linear in q, so precompute per-scenario totals:
    load_mwh = load_scenarios.sum(axis=1)                         # per scenario
    base_cost = (load_scenarios * price_scenarios).sum(axis=1)    # unhedged $ per scenario
    # per MW of contract k in scenario s: market value of its energy, and its MWh (200 × 3)
    hedge_value = np.column_stack([(price_scenarios * s).sum(axis=1) for s in shapes.values()])
    hedge_mwh = np.column_stack(
        [np.broadcast_to(s, price_scenarios.shape).sum(axis=1) for s in shapes.values()]
    )

    mo.vstack([mo.md("## Code (1/2): the fixed-price contracts and their energy per scenario"), mo.md(r"<small>**Inputs** (the scenarios from the risk slides) &ensp;|&ensp; `price_scenarios`: 200 × 720 array, scenario × hour of a 30-day month, DLAP price (\$/MWh) &ensp;|&ensp; `load_scenarios`: same shape, LSE load (MW) &ensp;|&ensp; `solar_scenarios`: same shape, solar output per MW of PPA (≈0–1) &ensp;|&ensp; `hour_of_day`: 720 hours’ clock hour, 0–23</small>"), mo.show_code(), terms(r"**HE7–HE22**: hour ending 7 through 22, i.e. 6:00–22:00")])
    return (
        base_cost,
        default_caps,
        default_prices,
        fair_values,
        hedge_mwh,
        hedge_value,
        hedges,
        load_mwh,
    )


@app.cell
def _(
    base_cost,
    default_caps,
    default_prices,
    hedge_mwh,
    hedge_value,
    linprog,
    load_mwh,
    mo,
    np,
):
    def hedge_lp(cvar_limit, alpha=0.95, fixed_prices=None, caps=None, price_shift=0.0):
        """min E[cost] s.t. CVaR_α ≤ limit (Rockafellar–Uryasev); None if infeasible."""
        fixed_prices = default_prices if fixed_prices is None else np.asarray(fixed_prices)
        caps = default_caps if caps is None else caps
        saving = hedge_value + (price_shift - fixed_prices) * hedge_mwh  # [s,k]: market − fixed, per MW
        cost = base_cost + price_shift * load_mwh                     # unhedged, per scenario
        S, K = saving.shape                                           # variables: q (K), z, η (S)
        c = np.r_[-saving.mean(axis=0), 0.0, np.zeros(S)]            # E[cost] = E[market] − E[saving]·q
        A_ub = np.vstack([
            np.hstack([-saving, -np.ones((S, 1)), -np.eye(S)]),       # η_s ≥ C_s(q) − z
            np.r_[np.zeros(K), 1.0, np.full(S, 1 / ((1 - alpha) * S))],  # z + Ση/((1−α)S) ≤ R̄
        ])
        bounds = [(0, cap) for cap in caps] + [(None, None)] + [(0, None)] * S
        res = linprog(c, A_ub=A_ub, b_ub=np.r_[-cost, cvar_limit * 1e6], bounds=bounds,
                      method="highs")
        if res.status != 0:
            return None
        q = res.x[:K]
        # sensitivity: the fixed price at which each contract's reduced cost is 0
        reduced_cost = res.lower.marginals[:K] + res.upper.marginals[:K]
        d_reduced_cost_d_price = hedge_mwh.mean(axis=0) - res.ineqlin.marginals[:S] @ hedge_mwh
        return {
            "q": q,
            "cost": (cost - saving @ q) / 1e6,                       # $M per scenario
            "risk_price": -res.ineqlin.marginals[-1],                # $ of E[cost] per $ of CVaR
            "accept_price": fixed_prices - reduced_cost / d_reduced_cost_d_price,
        }

    mo.vstack([mo.md("## Code (2/2): the CVaR hedging LP, its dual (price of risk) and the highest price we’d accept"), mo.show_code()])
    return (hedge_lp,)


@app.cell(hide_code=True)
def _(
    terms,
    cvar,
    cvar_slider,
    hedge_lp,
    hedges,
    make_subplots,
    mo,
    np,
    pd,
    show_chart,
    unhedged_cost,
):
    def hedging_slide(cvar_limit):
        solution = hedge_lp(cvar_limit)
        if solution is None:
            return mo.vstack([
                mo.md("## Risk vs. cost: the hedging frontier"), cvar_slider,
                mo.callout(mo.md(f"No hedge mix gets CVaR down to \\${cvar_limit:.2f}M: the LP is infeasible."),
                           kind="danger"),
            ])
        q, hedged_cost = solution["q"], solution["cost"]
        limits = np.linspace(cvar_slider.start, cvar(unhedged_cost), 25)
        frontier = [s["cost"] for s in map(hedge_lp, limits) if s is not None]

        fig = make_subplots(1, 2, horizontal_spacing=0.1,
                            subplot_titles=("", "Efficient frontier: one LP per CVaR limit"))
        fig.update_annotations(font_size=14)
        low = min(unhedged_cost.min(), hedged_cost.min())
        bins = dict(start=low, size=(unhedged_cost.max() - low) / 34 * (1 + 1e-9))  # 34 bins
        fig.add_histogram(x=unhedged_cost, xbins=bins, marker_color="#C9C2B0", name="Unhedged",
                          hovertemplate="$%{x}M: %{y} months<extra></extra>", row=1, col=1)
        fig.add_histogram(x=hedged_cost, xbins=bins, marker_color="#2F6DB5", opacity=0.8,
                          name=f"Hedged, CVaR ≤ ${cvar_limit:.2f}M",
                          hovertemplate="$%{x}M: %{y} months<extra></extra>", row=1, col=1)
        fig.add_vline(unhedged_cost.mean(), line=dict(color="#6A6A6A", width=2, dash="dash"),
                      showlegend=True, name=f"Unhedged average ${unhedged_cost.mean():.2f}M", row=1, col=1)
        fig.add_vline(hedged_cost.mean(), line=dict(color="#14213D", width=2, dash="dash"),
                      showlegend=True, name=f"Hedged average ${hedged_cost.mean():.2f}M", row=1, col=1)
        fig.update_xaxes(title="Monthly procurement cost ($M)", row=1, col=1)
        edges = low + bins["size"] * np.arange(35)
        peak = max(np.histogram(c, bins=edges)[0].max() for c in (unhedged_cost, hedged_cost))
        fig.update_yaxes(range=[0, 1.7 * peak], row=1, col=1)  # headroom for the legend

        fig.add_scatter(x=[cvar(c) for c in frontier], y=[c.mean() for c in frontier], mode="lines",
                        line=dict(color="#8A9BB8", width=2), showlegend=False,
                        hovertemplate="CVaR &#36;%{x:.2f}M: expected &#36;%{y:.2f}M<extra></extra>", row=1, col=2)
        fig.add_vline(cvar_limit, line=dict(color="#E73F74", width=1.5, dash="dash"), showlegend=True,
                      name=f"CVaR limit R\u0304 = ${cvar_limit:.2f}M", legend="legend2", row=1, col=2)
        fig.add_scatter(x=[cvar(hedged_cost)], y=[hedged_cost.mean()], mode="markers",
                        marker=dict(color="#E07A1F", size=15), name="Optimal hedge at this limit",
                        legend="legend2", row=1, col=2)
        fig.add_scatter(x=[cvar(unhedged_cost)], y=[unhedged_cost.mean()], mode="markers",
                        marker=dict(color="#3D3D3D", size=10, symbol="square"), showlegend=False,
                        hovertemplate="Unhedged<extra></extra>", row=1, col=2)
        fig.add_annotation(x=cvar(unhedged_cost), y=unhedged_cost.mean(), text="Unhedged",
                           showarrow=False, xanchor="right", yanchor="bottom", xshift=-6, yshift=6,
                           font=dict(size=12), row=1, col=2)
        fig.update_xaxes(title="CVaR 95% ($M)", row=1, col=2)
        fig.update_yaxes(title="Expected cost ($M)", row=1, col=2)

        mix = mo.md(
            "| Contract | Buys (MW) | Max MW | Fixed price |\n|:---|:---|:---|:---|\n"
            + "\n".join(f"| {name} | {mw:,.0f} | {cap} | \\${price:.2f} |"
                        for (name, (_, price, cap)), mw in zip(hedges.items(), q))
        )
        return mo.vstack(
            [
                mo.md("## Risk vs. cost: the hedging frontier"),
                cvar_slider,
                mo.hstack([show_chart(
                    fig, 340, barmode="overlay", bargap=0.05, margin=dict(l=10, r=10, t=30, b=10),
                    legend=dict(x=0.01, y=1, font=dict(size=11)),
                    legend2=dict(x=1, xanchor="right", y=1, font=dict(size=11),
                                 bgcolor="#F7F5EF", traceorder="normal"),
                ), mix], widths=[2.4, 1]),
                mo.md(
                    f"Expected cost **\\${hedged_cost.mean():.2f}M** (unhedged \\${unhedged_cost.mean():.2f}M), "
                    f"CVaR **\\${cvar(hedged_cost):.2f}M** (unhedged \\${cvar(unhedged_cost):.2f}M). "
                    "Every hedge is priced a little above its forecast value, so cutting tail risk "
                    "raises expected cost: the LP finds the cheapest mix for each risk limit.\n\n"
                    f"**Price of risk** (the dual of the CVaR limit): here, each \\$1 cut from CVaR costs "
                    f"**\\${solution['risk_price']:.2f}** of expected cost, the slope of the frontier."
                ),
            ]
        )

    mo.vstack([hedging_slide(cvar_slider.value), terms(r"$\bar R$: the CVaR limit (slider) &ensp;|&ensp; **fixed price** $K_k$: forecast value + premium &ensp;|&ensp; **price of risk**: dual of the CVaR row &ensp;|&ensp; **efficient frontier**: lowest expected cost at each risk level &ensp;|&ensp; **tail risk**: how bad the very bad outcomes are")])
    return


@app.cell(hide_code=True)
def _(
    cvar_slider,
    go,
    hedge_lp,
    hedge_mwh,
    hedges,
    load_mwh,
    mo,
    np,
    show_chart,
    unhedged_cost,
):
    def price_sensitivity_slide(cvar_limit):
        solution = hedge_lp(cvar_limit)
        if solution is None:
            return mo.md("## Price sensitivity\n\nNo feasible hedge at this CVaR limit.")
        q = solution["q"]
        shifts = np.arange(-20, 21, 1.0)

        # a fixed portfolio: re-price the same scenarios with every price shifted by Δ
        unhedged = [unhedged_cost.mean() + d * load_mwh.mean() / 1e6 for d in shifts]
        hedged = [
            (solution["cost"] + d * (load_mwh - hedge_mwh @ q) / 1e6).mean() for d in shifts
        ]
        open_mwh = (load_mwh - hedge_mwh @ q).mean()
        covered = 1 - open_mwh / load_mwh.mean()

        fig = go.Figure()
        fig.add_vline(0, line=dict(color="#C9C2B0", width=1))
        for y, color, name in [(unhedged, "#8A9BB8", "Unhedged"),
                               (hedged, "#2F6DB5", f"Hedged, CVaR ≤ ${cvar_limit:.2f}M")]:
            fig.add_scatter(x=shifts, y=y, mode="lines", line=dict(color=color, width=2.5), name=name,
                            hovertemplate="δλ = %{x:+.0f} &#36;/MWh: &#36;%{y:.2f}M<extra></extra>")
        fig.update_xaxes(title="Shift in every hourly price, δλ ($/MWh)")
        fig.update_yaxes(title="Expected monthly cost ($M)")

        rows = "\n".join(
            f"    | {name} | {mw:,.0f} MW | {(hedge_mwh[:, k] * mw).mean() / 1e3:,.0f} GWh |"
            for k, (name, mw) in enumerate(zip(hedges, q))
        )
        return mo.vstack(
            [
                mo.md(
                    "## Price sensitivity: how exposed is the LSE?\n\n"
                    "Shift every hourly price by δλ \\$/MWh and re-price the same portfolio. "
                    "The **slope** of each line is the MWh still exposed to the market price: "
                    "the **open position**."
                ),
                cvar_slider,
                mo.hstack(
                    [
                        show_chart(fig, 360, legend=dict(x=0.01, y=1)),
                        mo.md(
                            f"""
    | | Size | Energy per month |
    |:---|:---|:---|
    | Load | | {load_mwh.mean() / 1e3:,.0f} GWh |
{rows}
    | **Open position** | | **{open_mwh / 1e3:,.0f} GWh** |

    Unhedged, each \\$1/MWh move in prices changes cost by **\\${load_mwh.mean() / 1e6:.2f}M**.
    Hedged, by **\\${open_mwh / 1e6:.2f}M**: the hedges fix the price on **{covered:.0%}** of the energy.

    <small>**GWh**: 1,000 MWh &ensp;|&ensp; **Size**: MW bought of each contract at this CVaR limit</small>
    """
                        ),
                    ],
                    widths=[1.5, 1],
                ),
            ]
        )

    price_sensitivity_slide(cvar_slider.value)
    return


@app.cell(hide_code=True)
def _(terms, cvar_slider, fair_values, hedge_lp, hedges, mo):
    def acceptance_slide(cvar_limit):
        solution = hedge_lp(cvar_limit)
        if solution is None:
            return mo.md("## At what price would we accept an offer?\n\nNo feasible hedge at this CVaR limit.")
        rows = []
        for k, (name, (_, offered, cap)) in enumerate(hedges.items()):
            mw, accept = solution["q"][k], solution["accept_price"][k]
            if mw < 0.5:
                decision = f"Decline; counter at \\${accept:.2f} or less"
            elif mw > cap - 0.5:
                decision = f"Take all {cap} MW; would pay up to \\${accept:.2f}"
            else:
                decision = "Buying some: indifferent at the margin"
            rows.append(
                f"    | **{name}** | \\${fair_values[k]:.2f} | \\${offered:.2f} | {mw:,.0f} | "
                f"**\\${accept:.2f}** | {decision} |"
            )
        table = "\n".join(rows)

        return mo.vstack(
            [
                mo.md(
                    r"""
    ## At what price would we accept an offer?

    The solved LP answers this directly, in three steps:

    1. **Reduced cost** $\bar c_k$: how much expected cost would rise if we forced in one more MW of contract $k$.
       Positive: not worth buying at this price &ensp;|&ensp; zero: bought, and worth exactly its price &ensp;|&ensp; negative: we’d buy more if allowed.
    2. **Sensitivity** $\Delta_k$: how much $\bar c_k$ rises when the contract price rises \$1/MWh (the contract’s MWh, weighted for risk).
    3. **Highest acceptable price**: start from the offered price $K_k$ and move it until $\bar c_k = 0$: $\;K_k^{\star} = K_k - \bar c_k / \Delta_k$ is the most we’d pay.
    """
                ),
                mo.md(
                    f"""
    | Contract | Forecast value | Offered price $K_k$ | LSE buys (MW) | Accept at or below $K_k^\\star$ | Decision |
    |:---|:---|:---|:---|:---|:---|
{table}

    <small>At CVaR ≤ \\${cvar_limit:.2f}M; prices in \\$/MWh. $\\Delta_k = \\mathbb{{E}}[\\text{{MWh}}_k] + \\sum_s |y_s|\\,\\text{{MWh}}_{{sk}}$ ($\\text{{MWh}}_{{sk}}$: in scenario $s$).
    $y_s$: dual of the CVaR LP’s row $\\eta_s \\ge C_s - z$ ($z$: free threshold, $\\eta_s$: excess): extra weight the risk limit puts on month $s$, nonzero only in the worst months, summing to the price of risk. Exact for small changes; next slide re-solves.</small>
    """
                ),
            ]
        )

    mo.vstack([acceptance_slide(cvar_slider.value), terms(r"$\mathbb{E}$: average over the 200 scenarios &ensp;|&ensp; $\text{MWh}_k$: energy 1 MW of contract $k$ delivers &ensp;|&ensp; **forecast value**: mean price over those MWh &ensp;|&ensp; **counter**: offer a lower price")])
    return


@app.cell(hide_code=True)
def _(
    contract_dropdown,
    cvar_slider,
    fair_values,
    hedge_lp,
    hedges,
    mo,
    np,
    show_chart,
):
    def bid_curve_slide(contract, cvar_limit):
        k = list(hedges).index(contract)
        base_prices = np.array([price for _, price, _ in hedges.values()])
        premiums = np.arange(-5, 10.01, 0.1)

        bought = []
        for premium in premiums:
            prices = base_prices.copy()
            prices[k] = fair_values[k] + premium
            solution = hedge_lp(cvar_limit, fixed_prices=prices)
            bought.append(np.nan if solution is None else solution["q"][k])
        at_offer = hedge_lp(cvar_limit)
        accept_premium = (
            None if at_offer is None else at_offer["accept_price"][k] - fair_values[k]
        )
        offered_premium = base_prices[k] - fair_values[k]

        fig = go.Figure(go.Scatter(
            x=premiums, y=bought, mode="lines", line=dict(shape="hv", color="#2F6DB5", width=2.5),
            name="Re-solving the LP at each price",
            hovertemplate="+$%{x:.1f}/MWh: %{y:,.0f} MW<extra></extra>",
        ))
        fig.add_vline(offered_premium, line=dict(color="#8A9BB8", dash="dot", width=1.5),
                      showlegend=True, name="Offered price")
        if accept_premium is not None:
            fig.add_vline(accept_premium, line=dict(color="#E73F74", dash="dash", width=2),
                          showlegend=True, name="Highest acceptable price, from the solved LP")
        fig.update_xaxes(title="Fixed price above forecast value ($/MWh)")
        fig.update_yaxes(title="MW the LSE buys")

        return mo.vstack(
            [
                mo.md(
                    "## Negotiating a bilateral contract: the bid curve\n\n"
                    "Re-solve the hedging LP at many fixed prices for one contract (the others stay available). "
                    "The MW bought at each price is the LSE’s **bid curve**; the dashed line is the highest "
                    "acceptable price read off the single solved LP. Gaps mean no mix meets the CVaR limit."
                ),
                mo.hstack([contract_dropdown, cvar_slider], justify="start", gap=2),
                show_chart(fig, 320, margin=dict(l=10, r=10, t=40, b=10),
                           title=dict(text=f"LSE’s bid curve for the {contract}", font=dict(size=14), x=0.5),
                           legend=dict(x=1, xanchor="right", y=1)),
                mo.md(
                    "A seller has its own floor (for a solar developer, its cost of building and running "
                    "the plant). **Any price between the seller’s floor and the LSE’s highest acceptable price "
                    "beats no deal for both sides**; the LP tells the LSE where its end of that range is. "
                    "Tightening the CVaR limit raises that price, but only slightly here (cents, for solar): "
                    "these contracts add little risk reduction at the margin."
                ),
            ]
        )

    bid_curve_slide(contract_dropdown.value, cvar_slider.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Additional topics

    - **Uncertainty.** Wind and solar forecasts err. ISOs still run *deterministic* unit commitment and
      cover the error with reserve and ramping products; *stochastic* and *robust* unit commitment, which
      optimize across scenarios, remain mostly research. In planning, **two-stage stochastic programs** (invest now,
      dispatch per scenario) are common: see PyPSA’s [stochastic optimization example](https://docs.pypsa.org/latest/examples/stochastic-optimization/).
    - **Batteries.** State of charge links every hour to the next, and a battery’s real cost is the
      *opportunity cost* of energy it could sell later, so how storage should bid is an open question.
      With round-trip losses an LP may also charge and discharge at once to burn energy when prices are
      negative; models forbid it with a binary or accept it as harmless.
    - **Pricing nonconvexity.** How to price a MILP is still contested: MISO uses ELMP; PJM, NYISO, ISO-NE and SPP
      use fast-start pricing rules; convex hull pricing is the benchmark, but exact versions are hard to compute.

    <small>**Terms** — **state of charge**: energy stored in a battery &ensp;|&ensp; **round-trip losses**: energy lost charging and discharging (10–15%) &ensp;|&ensp; **ramping products**: reserves for fast MW changes &ensp;|&ensp; **fast-start pricing**: lets quick-start units’ commitment costs set the price &ensp;|&ensp; **deterministic / stochastic / robust UC**: one forecast / averaged over scenarios / worst case over a range</small>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Summary: the LP–market dictionary

    | LP concept | Electricity market meaning |
    |:---|:---|
    | Objective | Total production cost (welfare) |
    | Equality constraint | Supply = demand at each bus |
    | Inequality constraints | Generator limits, line limits, reserve requirements |
    | Dual variable | Price (LMP), congestion, reserve and capacity prices; price of risk |
    | Complementary slackness | Which generator sets the price |
    | Integer variables | On/off commitment decisions |
    | No strong duality (MILP) | Uplift payments and pricing debates |
    | CVaR linearization | Hedging an LSE’s procurement cost against price scenarios |
    | Reduced cost | Highest price an LSE should accept for a contract |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix A · Glossary: markets and institutions (1/4)

    | Term | Meaning |
    |:---|:---|
    | **ISO / RTO** | Independent System Operator / Regional Transmission Organization: the nonprofit that runs the grid and the wholesale market in a region (CAISO, ERCOT, MISO, PJM, SPP, NYISO, ISO-NE). |
    | **BA** (balancing authority) | The entity responsible for keeping supply equal to demand within its area. Every ISO is a BA; outside ISOs, utilities are their own BAs. |
    | **LSE** (load-serving entity) | A utility or retail supplier that buys wholesale power to serve its customers. |
    | **Offer / bid** | A generator’s offer to sell (MW at \$/MWh); a buyer’s bid to buy. |
    | **Award** | The MW a generator is told to produce after the market clears. |
    | **Clearing price** | The single price paid to every awarded MWh: the dual of supply = demand. |
    | **Merit order / marginal unit** | Offers stacked cheapest first; the marginal unit is the last one needed, and it sets the price. |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix A · Glossary: markets and institutions (2/4)

    | Term | Meaning |
    |:---|:---|
    | **LMP** (locational marginal price) | The price at one location (bus): the dual of that bus’s balance constraint. Energy + congestion + losses. |
    | **Congestion / congestion rent** | A line at its limit separates prices; the ISO collects the price difference × flow. |
    | **FTR / CRR / TCC** | Financial transmission right (CAISO/ERCOT: congestion revenue right; NYISO: transmission congestion contract): pays its holder MW × (LMP at sink − LMP at source), funded by congestion rent. |
    | **Hub / load zone / DLAP** | Price points averaged over many buses: trading hubs, zones where load settles, CAISO’s *default load aggregation points*. |
    | **Day-ahead market** | Cleared the day before, hourly; commits units (MILP). |
    | **Real-time market** | Cleared every 5 minutes (ERCOT, PJM, MISO and SPP call it SCED, security-constrained economic dispatch); adjusts output (LP). |
    | **Settlement** | Paying out: MWh × price, per hour and location. |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix A · Glossary: markets and institutions (3/4)

    | Term | Meaning |
    |:---|:---|
    | **Uplift / make-whole** | Payment outside the price to a unit whose market revenue doesn’t cover its costs. |
    | **Reserves** | Capacity held back to cover a sudden outage; has its own price. |
    | **Scarcity pricing / ORDC** | Higher prices as reserves run short; ERCOT uses an *operating reserve demand curve*. |
    | **Resource adequacy (RA)** | Having enough capacity for the peak hour years ahead. |
    | **Capacity market** | An auction (PJM, MISO, ISO-NE, NYISO) buying capacity (MW-day), not energy (MWh). California’s RA is bilateral, not an auction. |
    | **Energy-only market** | No capacity product; scarcity prices pay for capacity (ERCOT). |
    | **Missing money** | The fixed costs a plant can’t recover from energy prices alone. |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix A · Glossary: markets and institutions (4/4)

    | Term | Meaning |
    |:---|:---|
    | **CONE / Net CONE** | Cost of new entry for a new plant; *net* of its expected energy and reserve revenue. |
    | **VRR curve** | PJM’s capacity demand curve (Variable Resource Requirement). |
    | **LOLE, “1 day in 10 years”** | Loss-of-load expectation: the reliability target behind the capacity requirement. |
    | **Demand response** | Consumers paid to cut load when called. |
    | **Fast-start pricing / ELMP** | Rules letting quick-start units’ startup and no-load costs set the price; ELMP is MISO’s *extended LMP*. |
    | **WEIM** | CAISO’s Western Energy Imbalance Market: real-time dispatch shared with other western BAs. |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix B · Glossary: grid and generation (1/3)

    | Term | Meaning |
    |:---|:---|
    | **MW vs. MWh** | MW is a rate (power); MWh is an amount (energy). 1 MW for 1 hour = 1 MWh. |
    | **\$/MWh, \$/MW-day** | Energy prices; capacity prices (per MW of capacity, per day). |
    | **Bus** | A node in the network model: a substation at one voltage. |
    | **Line, thermal limit** | A transmission connection; the most power it can carry before overheating ($F_\ell$). |
    | **Reactance** ($x_\ell$) | A line’s opposition to AC power flow; power splits across paths in inverse proportion to it. |
    | **kV** | Kilovolts. Higher-voltage lines (345–500 kV) move bulk power between regions. |
    | **Interconnection** | A synchronized grid: Eastern, Western (WECC), Texas (ERCOT). |
    | **HVDC** | High-voltage direct current: a line whose flow is set by the operator, not by physics. |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix B · Glossary: grid and generation (2/3)

    | Term | Meaning |
    |:---|:---|
    | **AC vs. DC power flow** | AC: the true, nonconvex physics. DC: a linear approximation that keeps OPF an LP. |
    | **OPF** | Optimal power flow: dispatch that minimizes cost subject to network limits. |
    | **PTDF** | Power transfer distribution factor: the share of an injection that flows on each line. |
    | **Loop flow** | Power taking parallel paths it wasn’t “sent” on. |
    | **Per-unit** | Quantities scaled by a base (e.g., 100 MVA) in network models. |
    | **Dispatchable** | Output can be set by the operator, between minimum $\underline P$ and maximum $\overline P$ (gas, coal, hydro, batteries). |
    | **Curtailment** | Using less wind or solar than the weather provides; the only control over non-dispatchable output. |
    | **Commitment** | Deciding which units are on. Once on, a unit runs between its minimum and maximum. |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix B · Glossary: grid and generation (3/3)

    | Term | Meaning |
    |:---|:---|
    | **Startup / no-load cost** | Fixed cost to start a unit / to keep it running at all, independent of output. |
    | **Minimum up/down time** | Hours a unit must stay on (off) after starting (stopping). |
    | **Ramp rate** | How fast a unit can change output. |
    | **Baseload / peaker** | Units that run most hours / that run only at peaks (expensive fuel, cheap to build). |
    | **CC / CT** | Gas combined-cycle (efficient) / combustion turbine (a typical peaker). |
    | **Capacity factor** | Average output ÷ maximum output. |
    | **Duck curve** | Midday price/net-load dip from solar, then a steep evening ramp. |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix C · Glossary: optimization and risk (1/3)

    | Term | Meaning |
    |:---|:---|
    | **LP / MILP** | Linear program / mixed-integer linear program (some variables restricted to integers, here 0/1). |
    | **Dual variable / shadow price** | Change in optimal cost per unit relaxation of a constraint. |
    | **KKT conditions** | Optimality conditions: feasibility, dual feasibility, stationarity, complementary slackness. |
    | **Complementary slackness** | A constraint’s dual is zero unless the constraint binds. |
    | **Strong duality** | Primal and dual optimal values are equal (zero duality gap). Holds for LPs. |
    | **Lagrangian relaxation** | Moving a constraint into the objective with a price; for markets, it splits the problem by generator. |
    | **Duality gap** | Primal optimum minus dual optimum; generally positive for MILPs. |
    | **Convex hull** | The smallest convex set containing a set; its lower boundary is the best convex underestimate of a cost curve. |
    | **LP relaxation** | Dropping integrality ($u \in \{0,1\} \to [0,1]$). |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix C · Glossary: optimization and risk (2/3)

    | Term | Meaning |
    |:---|:---|
    | **Branch-and-bound** | The MILP algorithm: solve LP relaxations, branch on fractional variables, prune by bounds. |
    | **Optimality gap** | How far the best solution found may be from optimal; MILP solves stop at a target gap. |
    | **Min-cost network flow / transshipment** | LP that ships flow from sources to sinks over capacitated arcs at least cost, with flow conservation at every node. |
    | **Side constraint** | Any constraint added to a network-flow LP beyond flow conservation and arc capacities; it breaks the pure network structure (e.g., KVL). |
    | **Kirchhoff’s voltage law (KVL)** | Around any loop, reactance-weighted flows sum to zero; the side constraint that makes DC-OPF more than a network flow. |
    | **Total unimodularity** | Property of network-flow constraint matrices that guarantees integral LP solutions; KVL rows break it. |
    | **Reduced cost** | How much the objective worsens per unit of a variable forced into the solution; zero for variables strictly between their bounds. |
    | **Highest acceptable price** | The fixed price at which a contract’s reduced cost is zero: the most the LSE should pay for one more MW. |
    | **Open position** | Energy still bought at the market price after hedges; the slope of cost with respect to price. |
    | **Hedge** | A contract whose payoff offsets a cost risk. |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix C · Glossary: optimization and risk (3/3)

    | Term | Meaning |
    |:---|:---|
    | **Forward** | Agreement now to buy energy later at a fixed price; **on-peak** = 6:00–22:00 in the West, 7:00–23:00 in the East (Mon–Sat in the West, Mon–Fri in the East). |
    | **Fixed-price contract** | Energy bought ahead in a set delivery shape at a fixed price; the rest of the load is bought at market. |
    | **Contract price / premium** | The fixed price paid / how much it exceeds the forecast market value. |
    | **PPA** | Power purchase agreement: buying a specific plant’s output, e.g., a solar farm’s. |
    | **Scenario / bootstrap** | One possible future / building scenarios by resampling historical days. |
    | **Stochastic programming** | Decide now, then score the decision across scenarios of the future (expected cost, CVaR). Our hedging LP is one. |
    | **VaR / CVaR** | Value-at-risk: the cost exceeded only $(1-\alpha)$ of the time / conditional VaR: the average cost in that worst $(1-\alpha)$. |
    | **Efficient frontier** | Best expected cost for each risk level. |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix D · Data sources

    | Source | What it is | Used for |
    |:---|:---|:---|
    | [**EIA-860**](https://www.eia.gov/electricity/data/eia860/) | U.S. Energy Information Administration’s annual survey of every generator ≥ 1 MW: location, capacity, fuel, balancing authority (2025 data). | Power plant, ISO and WECC maps |
    | [**EIA-930**](https://www.eia.gov/electricity/gridmonitor/) | EIA’s hourly balancing-authority data: demand, generation by fuel, interchange. No prices. | PG&E-area load, CAISO solar shape |
    | [**PUDL**](https://catalyst.coop/pudl/) | Public Utility Data Liberation, by Catalyst Cooperative: cleaned, open EIA and FERC data. How we accessed EIA-860 and EIA-930. | All EIA data |
    | [**CAISO OASIS**](https://oasis.caiso.com/) | CAISO’s public market data: day-ahead LMPs with energy, congestion and loss parts. Keeps about 3 years. | PG&E DLAP prices, Jan 2024 – Oct 2026 |
    | [**ERCOT market data**](https://www.ercot.com/mktinfo/rtm) | Real-time [LMP contour map](https://www.ercot.com/content/cdr/contours/rtmLmp.html) and [settlement point prices](https://www.ercot.com/content/cdr/html/real_time_spp.html). | Live LMP map, Texas price spread |
    | [**GridSFM US Power Grid**](https://huggingface.co/datasets/microsoft/GridSFM_US_power_grid) | Microsoft Research’s open, approximate transmission models built from OpenStreetMap, EIA and Census data (MIT license). | Western transmission map and DC-OPF |
    | [**PyPSA example: Negative LMPs from Line Congestion**](https://docs.pypsa.org/latest/examples/negative-prices-kvl-baker/) | After [K. Baker, *3bus_LMPs*](https://github.com/kyribaker/3bus_LMPs) and [Baker & Gangammanavar (2024)](https://doi.org/10.48550/arXiv.2403.19032). | Negative-price three-bus example |
    | [**Census cartographic boundaries**](https://www.census.gov/geographies/mapping-files/time-series/geo/cartographic-boundary.html) | U.S. state outlines (2023, 1:20M). | Map backgrounds |

    All data is downloaded by `scripts/build_data.py` and stored in `assets/`, so the talk runs offline.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix E · Software and references

    | Software | What it is |
    |:---|:---|
    | [**PyPSA**](https://pypsa.org) ([docs](https://docs.pypsa.org), [GitHub](https://github.com/PyPSA/PyPSA)) | Python for Power System Analysis: open-source power-system modeling. Builds the dispatch, DC-OPF and unit-commitment models here. |
    | [**HiGHS**](https://highs.dev) | Open-source LP/MILP solver used by PyPSA and by `scipy.optimize.linprog`. |
    | [**linopy**](https://github.com/PyPSA/linopy) | The optimization modeling layer under PyPSA. |
    | [**SciPy `linprog`**](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) | Plain-matrix LP interface: capacity auction and hedging LPs. |
    | [**marimo**](https://marimo.io) | Reactive Python notebooks; this deck is one, presented in slides layout. |

    **This deck:** notebook, data and build scripts at [github.com/jleighfields/electricity_market_optimization](https://github.com/jleighfields/electricity_market_optimization).

    **Further reading:** J. A. Taylor, *Convex Optimization of Power Systems*, Cambridge University Press, 2015 &ensp;|&ensp;
    D. S. Kirschen and G. Strbac, *Fundamentals of Power System Economics*, 2nd ed., Wiley, 2019.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix F · Agencies

    | Agency | Role |
    |:---|:---|
    | **FERC** | Federal Energy Regulatory Commission: regulates wholesale markets and interstate transmission. |
    | **CPUC** | California Public Utilities Commission: sets California’s resource adequacy requirements. |
    | **NERC** | North American Electric Reliability Corporation: sets reliability standards. |
    | **WECC** | Western Electricity Coordinating Council: reliability region for the Western Interconnection. |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Appendix G · Live price maps

    Every ISO publishes its current prices; most show them as a map that updates every 5 minutes.

    | Market | Live map |
    |:---|:---|
    | **All US ISOs** | [GridStatus nodal price map](https://www.gridstatus.io/map): every pricing node on one map (free account required) |
    | **ERCOT** (Texas) | [Real-time LMP contour map](https://www.ercot.com/content/cdr/contours/rtmLmp.html), the map on our ERCOT slide |
    | **CAISO** and the western market | [Today’s Outlook: prices](https://www.caiso.com/todays-outlook/prices) |
    | **SPP** (incl. RTO West) | [Price contour map](https://pricecontourmap.spp.org/pricecontourmap/) |
    | **MISO** | [Real-time LMP contour map](https://api.misoenergy.org/MISORTWD/lmpcontourmap.html) |
    | **PJM** | [Interregional data map](https://www.pjm.com/markets-and-operations/interregional-map) (PJM and its neighbors) |
    | **NYISO** | [Real-time dashboard](https://www.nyiso.com/real-time-dashboard) |
    | **ISO-NE** | [ISO Express real-time maps](https://www.iso-ne.com/isoexpress/) |

    <small>Look for congestion: a sharp color change across a short distance is a binding line, the μ of our two-bus example.</small>
    """
    )
    return


if __name__ == "__main__":
    app.run()
