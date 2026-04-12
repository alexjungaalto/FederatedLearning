#!/usr/bin/env python3
"""
FML2ILV_Privacy_FMI.py
----------------------
Privacy demo for FML2ILV Session 6: Privacy in Federated Learning.

Illustrates that sharing detailed local data statistics (2D histograms of
tmin vs tmax) can identify a weather station, while sharing only aggregate
statistics (mean tmin, mean tmax) makes identification much harder.

Steps:
  1. Load FMI station list and subsample ~10% evenly via k-means on lat/lon.
  2. Download daily tmin/tmax for January 2024 from all selected stations.
  3. Plot: 2D histograms (fingerprints) for each station — unique patterns.
  4. Plot: aggregate stats (mean tmin, mean tmax) — many stations overlap.

Usage:
    python FML2ILV_Privacy_FMI.py
"""

import os
import requests
import xml.etree.ElementTree as ET
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# ── Configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
STATIONS_CSV  = os.path.join(SCRIPT_DIR, "fmi_stations.csv")
CACHE_FILE    = os.path.join(SCRIPT_DIR, "privacy_fmi_jan2024.csv")
START         = "2024-01-01"
END           = "2024-01-31"
SAMPLE_FRAC   = 0.10                      # use ~10% of stations
RANDOM_STATE  = 42
HIST_BINS     = 8                         # bins per axis for 2D histogram

# ── FMI WFS XML parser (reused from Session 3 demos) ────────────────────────

NS = {
    "wfs":    "http://www.opengis.net/wfs/2.0",
    "wml2":   "http://www.opengis.net/waterml/2.0",
    "gml":    "http://www.opengis.net/gml/3.2",
    "target": "http://xml.fmi.fi/namespace/om/atmosphericfeatures/1.1",
    "om":     "http://www.opengis.net/om/2.0",
}

def _parse_fmi_xml(content):
    """Parse FMI WFS XML into a list of dicts."""
    root = ET.fromstring(content)
    rows = []
    for member in root.findall(".//wfs:member", NS):
        loc   = member.find(".//target:Location", NS)
        stn   = loc.find(
            './gml:name[@codeSpace='
            '"http://xml.fmi.fi/namespace/locationcode/name"]', NS).text
        href  = member.find(".//om:observedProperty", NS).get(
            "{http://www.w3.org/1999/xlink}href", "")
        param = href.split("param=")[1].split("&")[0] if "param=" in href else ""
        for pt in member.findall(".//wml2:point", NS):
            t = pt.find(".//wml2:time", NS).text
            if datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").hour != 0:
                continue
            v = pt.find(".//wml2:value", NS).text
            rows.append({
                "station": stn, "day": t[:10], "parameter": param,
                "value": float(v) if v != "NaN" else None,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = (df.pivot_table(index=["station", "day"], columns="parameter",
                         values="value", aggfunc="first").reset_index())
    df.columns.name = None
    return df


# ── Step 1: Load station list and subsample via k-means ─────────────────────

print("Step 1: Loading FMI stations and subsampling …")
stations = pd.read_csv(STATIONS_CSV)
coords   = stations[["lat", "lon"]].values

n_sample = max(3, int(len(coords) * SAMPLE_FRAC))
kmeans   = KMeans(n_clusters=n_sample, random_state=RANDOM_STATE, n_init=10)
kmeans.fit(coords)

# Pick the station closest to each cluster centroid
sample_idx = []
for k in range(n_sample):
    members = np.where(kmeans.labels_ == k)[0]
    dists   = cdist([kmeans.cluster_centers_[k]], coords[members])
    sample_idx.append(members[np.argmin(dists)])

selected = stations.iloc[sample_idx].reset_index(drop=True)
print(f"  {len(stations)} total stations → {len(selected)} selected ({SAMPLE_FRAC:.0%})")
for _, row in selected.iterrows():
    print(f"    {row['station']:40s}  ({row['lat']:.2f}, {row['lon']:.2f})")


# ── Step 2: Download daily tmin/tmax for selected stations ──────────────────

if os.path.exists(CACHE_FILE):
    print(f"\nStep 2: Loading cached data from {CACHE_FILE}")
    df_all = pd.read_csv(CACHE_FILE)
else:
    print(f"\nStep 2: Fetching tmin/tmax for {START} → {END} …")
    frames = []
    for i, row in selected.iterrows():
        place = row["station"].split()[0]  # use first word as place query
        url = (
            "http://opendata.fmi.fi/wfs?service=WFS&version=2.0.0"
            "&request=getFeature"
            "&storedquery_id=fmi::observations::weather::daily::timevaluepair"
            f"&place={place}"
            f"&starttime={START}T00:00:00Z"
            f"&endtime={END}T00:00:00Z"
            "&parameters=tmin,tmax&timestep=720"
        )
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            df = _parse_fmi_xml(r.content)
            if not df.empty:
                frames.append(df)
                print(f"  [{i+1}/{len(selected)}] {row['station']}: "
                      f"{len(df)} days")
            else:
                print(f"  [{i+1}/{len(selected)}] {row['station']}: no data")
        except Exception as e:
            print(f"  [{i+1}/{len(selected)}] {row['station']}: ERROR {e}")

    df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df_all.empty:
        df_all.to_csv(CACHE_FILE, index=False)
        print(f"  Saved to {CACHE_FILE}")

df_all = df_all.dropna(subset=["tmin", "tmax"]).reset_index(drop=True)
station_names = df_all["station"].unique()
n_stations    = len(station_names)
print(f"  {len(df_all)} valid day-records across {n_stations} stations\n")

if n_stations == 0:
    print("No data — exiting.")
    raise SystemExit


# ── Step 3: 2D histograms — fingerprints that identify stations ─────────────

print("Step 3: Plotting 2D histograms (detailed fingerprints) …")

# Global bin edges so all histograms are comparable
tmin_edges = np.linspace(df_all["tmin"].min() - 1, df_all["tmin"].max() + 1, HIST_BINS + 1)
tmax_edges = np.linspace(df_all["tmax"].min() - 1, df_all["tmax"].max() + 1, HIST_BINS + 1)

ncols = min(4, n_stations)
nrows = int(np.ceil(n_stations / ncols))

fig1, axes1 = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows),
                            squeeze=False)
fig1.suptitle("2D Histograms of (tmin, tmax) — Each Station Has a Unique Fingerprint",
              fontsize=13, fontweight="bold")

for idx, stn in enumerate(station_names):
    ax  = axes1[idx // ncols, idx % ncols]
    sub = df_all[df_all["station"] == stn]
    ax.hist2d(sub["tmin"], sub["tmax"],
              bins=[tmin_edges, tmax_edges], cmap="Blues", cmin=0.5)
    ax.set_title(stn.split()[0], fontsize=9)
    ax.set_xlabel("tmin (°C)", fontsize=7)
    ax.set_ylabel("tmax (°C)", fontsize=7)
    ax.tick_params(labelsize=6)

# Hide unused subplots
for idx in range(n_stations, nrows * ncols):
    axes1[idx // ncols, idx % ncols].set_visible(False)

fig1.tight_layout(rect=[0, 0, 1, 0.94])
fig1.savefig("privacy_2d_histograms.png", dpi=120, bbox_inches="tight")
print("  Saved → privacy_2d_histograms.png")


# ── Step 4: Aggregate stats — stations overlap, harder to identify ──────────

print("Step 4: Plotting aggregate statistics (mean tmin, mean tmax) …")

agg = (df_all.groupby("station")[["tmin", "tmax"]]
       .agg(["mean", "std"]).reset_index())
agg.columns = ["station", "tmin_mean", "tmin_std", "tmax_mean", "tmax_std"]

fig2, (ax_detail, ax_agg) = plt.subplots(1, 2, figsize=(13, 5))

# Left: all daily (tmin, tmax) points colored by station — clearly separable
colors = plt.cm.tab20(np.linspace(0, 1, n_stations))
for idx, stn in enumerate(station_names):
    sub = df_all[df_all["station"] == stn]
    ax_detail.scatter(sub["tmin"], sub["tmax"], s=15, alpha=0.6,
                      color=colors[idx], label=stn.split()[0])
ax_detail.set_xlabel("tmin (°C)")
ax_detail.set_ylabel("tmax (°C)")
ax_detail.set_title("Raw daily data → stations identifiable", fontsize=10)
ax_detail.grid(True, alpha=0.25)
if n_stations <= 12:
    ax_detail.legend(fontsize=6, loc="upper left", ncol=2)

# Right: only mean ± std — many stations overlap
for idx, row in agg.iterrows():
    ax_agg.errorbar(row["tmin_mean"], row["tmax_mean"],
                    xerr=row["tmin_std"], yerr=row["tmax_std"],
                    fmt="o", color=colors[idx], markersize=6, capsize=3,
                    alpha=0.7, label=row["station"].split()[0])
ax_agg.set_xlabel("mean tmin (°C)")
ax_agg.set_ylabel("mean tmax (°C)")
ax_agg.set_title("Aggregate stats (mean ± std) → stations overlap", fontsize=10)
ax_agg.grid(True, alpha=0.25)
if n_stations <= 12:
    ax_agg.legend(fontsize=6, loc="upper left", ncol=2)

fig2.suptitle("Privacy Perspective: Detailed vs. Aggregate Statistics",
              fontsize=13, fontweight="bold")
fig2.tight_layout(rect=[0, 0, 1, 0.93])
fig2.savefig("privacy_detail_vs_aggregate.png", dpi=120, bbox_inches="tight")
print("  Saved → privacy_detail_vs_aggregate.png")

plt.show()
print("\nDone.")