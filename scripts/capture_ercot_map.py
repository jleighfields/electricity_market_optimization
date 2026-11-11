"""Keep the most congested ERCOT real-time LMP map seen, for the deck's ERCOT slide.

ERCOT only publishes the *current* contour map, so this script watches it: every few
minutes it reads the real-time LMP at all ~1,100 settlement points, and whenever the
nodal price gradient (98th minus 2nd percentile, so one odd node can't dominate) beats
the best so far it saves the map, legend and details (with hub/zone prices for the slide).

    uv run python scripts/capture_ercot_map.py                 # check once
    uv run python scripts/capture_ercot_map.py --hours 8       # watch for 8 hours

Writes to assets/: ercot_best_map.png, ercot_best_legend.png, ercot_best.json
"""

import argparse
import html
import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "assets"
PRICES = "https://www.ercot.com/content/cdr/html/real_time_spp.html"
NODES = "https://www.ercot.com/content/cdr/html/current_np6788.html"
MAP = "https://www.ercot.com/content/cdr/contours/rtmLmp.png"
LEGEND = "https://www.ercot.com/content/cdr/contours/rtmLmpLegend.png"
USER_AGENT = "electricity-market-optimization-data"


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def table_rows(url):
    page = fetch(url).decode("utf-8", errors="replace")
    return [[html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
             for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)]
            for row in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S)]


def nodal_spread():
    """98th minus 2nd percentile of the current real-time LMP across settlement points."""
    prices = []
    for row in table_rows(NODES):
        try:
            prices.append(float(row[1]))
        except (IndexError, ValueError):
            pass
    prices.sort()
    pick = lambda q: prices[round(q * (len(prices) - 1))]
    return pick(0.98) - pick(0.02), prices[0], prices[-1]


def latest_prices():
    """Latest interval's prices at ERCOT's hubs and load zones."""
    rows = table_rows(PRICES)
    header, last = rows[0], rows[-1]
    prices = {name: float(value) for name, value in zip(header[2:], last[2:])}
    return f"{last[0]} interval ending {last[1]}", prices


def check_once():
    interval, prices = latest_prices()
    low, high = min(prices, key=prices.get), max(prices, key=prices.get)
    spread = prices[high] - prices[low]
    nodal, node_min, node_max = nodal_spread()
    best_file = ASSETS / "ercot_best.json"
    best = json.loads(best_file.read_text()).get("nodal_spread", -1.0) if best_file.exists() else -1.0
    stamp = (f"{datetime.now():%H:%M} {interval}: nodal spread ${nodal:.2f} (min {node_min:.2f}, max {node_max:.2f}); "
             f"hub/zone {low} {prices[low]:.2f} → {high} {prices[high]:.2f}")
    if nodal > best:
        (ASSETS / "ercot_best_map.png").write_bytes(fetch(MAP))
        (ASSETS / "ercot_best_legend.png").write_bytes(fetch(LEGEND))
        best_file.write_text(json.dumps({
            "interval": interval, "spread": spread,
            "nodal_spread": nodal, "node_min": node_min, "node_max": node_max,
            "low": [low, prices[low]], "high": [high, prices[high]],
            "captured": f"{datetime.now():%Y-%m-%d %H:%M}",
        }, indent=2))
        print(stamp + "  ← new best, saved", flush=True)
    else:
        print(stamp + f"  (best ${best:.2f})", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=0, help="keep watching this long")
    parser.add_argument("--every", type=int, default=300, help="seconds between checks")
    args = parser.parse_args()
    end = time.time() + args.hours * 3600
    while True:
        try:
            check_once()
        except Exception as error:  # network hiccup: try again next round
            print(f"{datetime.now():%H:%M} check failed: {error}", flush=True)
        if time.time() + args.every > end:
            break
        time.sleep(args.every)
