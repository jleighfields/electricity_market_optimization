"""Download and slim the data behind the talk, so it runs offline.

    uv run python scripts/build_data.py            # everything
    uv run python scripts/build_data.py pge        # just one step

Writes to assets/:
  plants.parquet            EIA-860 (via PUDL) operating plants, one row per plant
  us_states.geojson         Census cartographic state outlines, lower 48
  western_16h_model.json    Microsoft GridSFM western interconnection model (peak hour)
  western_16h_dc_results.json  GridSFM's own DC-OPF solution (reference, for comparison)
  pge_hourly.parquet        CAISO day-ahead PG&E DLAP prices + EIA-930 PG&E load
                            and CAISO solar, hourly, Jan 2024 - Oct 2026
"""

import io
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

PUDL = "https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/stable/"
STATES = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"
GRIDSFM = (
    "https://huggingface.co/datasets/microsoft/GridSFM_US_power_grid/"
    "resolve/main/16h/western_model.json"
)
OASIS = "https://oasis.caiso.com/oasisapi/SingleZip"
USER_AGENT = "electricity-market-optimization-data"

YEAR = 2025  # latest EIA-860 year PUDL marks as final
NOT_LOWER_48 = {"AK", "HI", "PR", "VI", "GU", "AS", "MP"}
PRICE_START, PRICE_END = "2024-01-01", "2026-11-01"  # OASIS keeps ~3 years


def plants():
    cols = [
        "report_date", "plant_id_eia", "plant_name_eia", "state", "latitude",
        "longitude", "balancing_authority_code_eia", "operational_status",
        "capacity_mw", "fuel_type_code_pudl", "technology_description",
    ]
    g = pd.read_parquet(PUDL + "out_eia__yearly_generators.parquet", columns=cols)
    g = g[(pd.to_datetime(g.report_date).dt.year == YEAR) & (g.operational_status == "existing")]
    g = g[~g.state.isin(NOT_LOWER_48)].dropna(subset=["latitude", "longitude"])

    # primary fuel of a plant = fuel with the most capacity there
    by_fuel = g.groupby(["plant_id_eia", "fuel_type_code_pudl"])["capacity_mw"].sum()
    primary = by_fuel.reset_index().sort_values("capacity_mw").groupby("plant_id_eia").last()

    p = g.groupby("plant_id_eia").agg(
        plant_name=("plant_name_eia", "first"),
        state=("state", "first"),
        lat=("latitude", "first"),
        lon=("longitude", "first"),
        ba=("balancing_authority_code_eia", "first"),
        capacity_mw=("capacity_mw", "sum"),
    )
    p["fuel"] = primary["fuel_type_code_pudl"]
    p.reset_index().to_parquet(ASSETS / "plants.parquet", index=False)
    print(f"plants.parquet: {len(p):,} plants, {p.capacity_mw.sum() / 1e3:,.0f} GW")


def states():
    s = gpd.read_file(STATES)
    s = s[~s.STUSPS.isin(NOT_LOWER_48)][["STUSPS", "NAME", "geometry"]]
    s.to_file(ASSETS / "us_states.geojson", driver="GeoJSON")
    print(f"us_states.geojson: {len(s)} states")


def grid():
    urllib.request.urlretrieve(GRIDSFM, ASSETS / "western_16h_model.json")
    urllib.request.urlretrieve(GRIDSFM.replace("_model.json", "_dc_results.json"),
                               ASSETS / "western_16h_dc_results.json")
    print("western_16h_model.json, western_16h_dc_results.json: downloaded")


def oasis_month(start, end, node):
    """One month of CAISO day-ahead LMPs (and components) for one node.

    Each month is cached in assets/raw/caiso_oasis/, so reruns only fetch what's missing.
    """
    cache = ASSETS / "raw" / "caiso_oasis" / f"{node}_{start:%Y-%m}.csv"
    if cache.exists():
        return pd.read_csv(cache), True
    cache.parent.mkdir(parents=True, exist_ok=True)
    # OASIS allows at most 31 days per request, counted in Pacific time, so the
    # window must run midnight-to-midnight Pacific (07:00 or 08:00 UTC with DST).
    def utc(day):
        return pd.Timestamp(day, tz="America/Los_Angeles").tz_convert("UTC")

    params = {
        "queryname": "PRC_LMP", "version": 12, "market_run_id": "DAM", "node": node,
        "startdatetime": f"{utc(start):%Y%m%dT%H:%M}-0000",
        "enddatetime": f"{utc(end):%Y%m%dT%H:%M}-0000",
        "resultformat": 6,  # CSV
    }
    url = OASIS + "?" + urllib.parse.urlencode(params, safe=":")
    for attempt in range(4):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        csvs = [n for n in archive.namelist() if n.endswith(".csv")]
        if csvs:
            cache.write_bytes(archive.read(csvs[0]))
            return pd.read_csv(cache), False
        # no CSV: OASIS sent an XML error file; retry only if it looks transient
        error = archive.read(archive.namelist()[0]).decode(errors="replace")
        message = error.split("<m:ERR_DESC>")[-1].split("<")[0] if "ERR_DESC" in error else error[:200]
        if "31 days" in message or "No data" in message:
            raise RuntimeError(f"OASIS {start:%Y-%m}: {message}")
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"OASIS {start:%Y-%m}: {message}; rerun to resume")


def pge(node="DLAP_PGAE-APND"):
    months = pd.date_range(PRICE_START, PRICE_END, freq="MS")
    frames = []
    for start, end in zip(months[:-1], months[1:]):
        month, cached = oasis_month(start, end, node)
        frames.append(month)
        print(f"  OASIS {start:%Y-%m}: {len(month)} rows" + (" (cached)" if cached else ""))
        if not cached:
            time.sleep(20)  # OASIS throttles bursts of requests
    raw = pd.concat(frames)
    prices = (
        raw.assign(time=pd.to_datetime(raw.INTERVALSTARTTIME_GMT, utc=True))
        .pivot_table(index="time", columns="LMP_TYPE", values="MW", aggfunc="first")
        .rename(columns={"LMP": "price", "MCE": "energy", "MCC": "congestion", "MCL": "loss"})
        [["price", "energy", "congestion", "loss"]]
    )

    # EIA-930 (and PUDL's datetime_utc) label each hour by its END; OASIS labels by START.
    # Shift EIA-930 back one hour so price, load and solar describe the same hour.
    utc = lambda s: pd.to_datetime(s).dt.tz_localize("UTC") - pd.Timedelta(hours=1)
    demand = pd.read_parquet(
        PUDL + "core_eia930__hourly_subregion_demand.parquet",
        filters=[("balancing_authority_subregion_code_eia", "==", "PGAE")],
    )
    load = demand.assign(time=utc(demand.datetime_utc)).set_index("time")["demand_reported_mwh"]

    generation = pd.read_parquet(
        PUDL + "core_eia930__hourly_net_generation_by_energy_source.parquet",
        filters=[("balancing_authority_code_eia", "==", "CISO")],
    )
    solar = generation[generation.generation_energy_source.astype(str).str.startswith("solar")]
    solar = (solar.assign(time=utc(solar.datetime_utc))
             .groupby("time")["net_generation_adjusted_mwh"].sum())

    hourly = prices.join(load.rename("pge_load_mw")).join(solar.rename("caiso_solar_mw"))
    hourly = hourly.loc[PRICE_START:PRICE_END].dropna()
    hourly.index = hourly.index.tz_convert("America/Los_Angeles")
    hourly.to_parquet(ASSETS / "pge_hourly.parquet")
    print(f"pge_hourly.parquet: {len(hourly):,} hours, "
          f"{hourly.index.min():%Y-%m-%d} to {hourly.index.max():%Y-%m-%d}")


STEPS = {"plants": plants, "states": states, "grid": grid, "pge": pge}

if __name__ == "__main__":
    for name in sys.argv[1:] or STEPS:
        STEPS[name]()
