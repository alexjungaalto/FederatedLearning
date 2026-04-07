#!/usr/bin/env python3
"""
Federated Learning Demo: Two devices train local max-temperature predictors
from min-temperature data using the FMI open WFS API.

Device 1 (Rovaniemi, Jan–Feb 2025)  →  DecisionTreeRegressor
Device 2 (Åland / Mariehamn, Jul–Aug 2020)  →  LinearRegression
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# ── FMI scraping (adapted from ReadInDailyMaxMin.py) ────────────────────────

NS = {
    "wfs": "http://www.opengis.net/wfs/2.0",
    "wml2": "http://www.opengis.net/waterml/2.0",
    "gml": "http://www.opengis.net/gml/3.2",
    "target": "http://xml.fmi.fi/namespace/om/atmosphericfeatures/1.1",
    "om": "http://www.opengis.net/om/2.0",
}

def _parse_fmi_xml(xml_bytes: bytes) -> pd.DataFrame:
    root = ET.fromstring(xml_bytes)
    rows = []
    for member in root.findall(".//wfs:member", NS):
        location = member.find(".//target:Location", NS)
        station = location.find(
            './gml:name[@codeSpace="http://xml.fmi.fi/namespace/locationcode/name"]', NS
        ).text
        param = (
            member.find(".//om:observedProperty", NS)
            .get("{http://www.w3.org/1999/xlink}href")
            .split("param=")[1]
            .split("&")[0]
        )
        for point in member.findall(".//wml2:point", NS):
            t = point.find(".//wml2:time", NS).text
            # daily values are anchored at midnight UTC
            import datetime
            if datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").hour != 0:
                continue
            val_text = point.find(".//wml2:value", NS).text
            rows.append({
                "station": station,
                "day": t[:10],
                "parameter": param,
                "value": float(val_text) if val_text != "NaN" else None,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = (
        df.pivot_table(index=["station", "day"], columns="parameter",
                       values="value", aggfunc="first")
        .reset_index()
    )
    df.columns.name = None
    return df


def fetch_daily(place: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily tmin / tmax for a named FMI place.
    start / end format: 'YYYY-MM-DD'
    """
    url = (
        "http://opendata.fmi.fi/wfs?service=WFS&version=2.0.0"
        "&request=getFeature"
        "&storedquery_id=fmi::observations::weather::daily::timevaluepair"
        f"&place={place}"
        f"&starttime={start}T00:00:00Z"
        f"&endtime={end}T00:00:00Z"
        "&parameters=tmin,tmax"
        "&timestep=720"
    )
    print(f"  Fetching {place} ({start} → {end}) …")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return _parse_fmi_xml(r.content)


# ── Device definitions ───────────────────────────────────────────────────────

devices = [
    {
        "id": 1,
        "label": "Device 1 – Rovaniemi (Jan–Feb 2025)",
        "place": "Rovaniemi",
        "start": "2025-01-01",
        "end": "2025-02-28",
        "model": DecisionTreeRegressor(max_depth=3, random_state=42),
        "model_name": "Decision Tree (depth 3)",
        "color": "#1f77b4",
    },
    {
        "id": 2,
        "label": "Device 2 – Mariehamn/Åland (Jul–Aug 2020)",
        "place": "Mariehamn",
        "start": "2020-07-01",
        "end": "2020-08-31",
        "model": LinearRegression(),
        "model_name": "Linear Regression",
        "color": "#d62728",
    },
]


# ── Fetch, train, evaluate ───────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Federated Learning Demo: Local Model Training", fontsize=13, fontweight="bold")

results = []

for dev, ax in zip(devices, axes):
    print(f"\n{'='*55}")
    print(f"{dev['label']}")
    print(f"  Model : {dev['model_name']}")

    df = fetch_daily(dev["place"], dev["start"], dev["end"])

    if df.empty or "tmin" not in df.columns or "tmax" not in df.columns:
        print("  [!] No data returned – skipping.")
        continue

    df = df.dropna(subset=["tmin", "tmax"]).reset_index(drop=True)
    print(f"  Samples after cleaning: {len(df)}")

    X = df[["tmin"]].values
    y = df["tmax"].values

    dev["model"].fit(X, y)
    y_pred = dev["model"].predict(X)
    rmse = mean_squared_error(y, y_pred) ** 0.5

    print(f"  Train RMSE : {rmse:.2f} °C")
    results.append({"device": dev["label"], "model": dev["model_name"],
                    "n": len(df), "rmse": rmse})

    # ── Plot ────────────────────────────────────────────────────────────────
    AXIS_MIN, AXIS_MAX = -30, 30
    t_range = np.linspace(AXIS_MIN, AXIS_MAX, 200).reshape(-1, 1)
    ax.scatter(X, y, alpha=0.7, s=30, color=dev["color"], label="Observed", zorder=3)
    ax.plot(t_range, dev["model"].predict(t_range), color="k", lw=2,
            label=f"{dev['model_name']}\n(RMSE={rmse:.2f} °C)")
    ax.set_xlim(AXIS_MIN, AXIS_MAX)
    ax.set_ylim(AXIS_MIN, AXIS_MAX)
    ax.set_xlabel("Min temperature (°C)")
    ax.set_ylabel("Max temperature (°C)")
    ax.set_title(dev["label"], fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ── Summary table ────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print("Summary")
print(f"{'='*55}")
for r in results:
    print(f"  {r['device']}")
    print(f"    Model : {r['model']}")
    print(f"    n     : {r['n']}")
    print(f"    RMSE  : {r['rmse']:.2f} °C")

plt.tight_layout()
plt.savefig("fl_temp_demo.png", dpi=150, bbox_inches="tight")
print("\nFigure saved → fl_temp_demo.png")
plt.show()