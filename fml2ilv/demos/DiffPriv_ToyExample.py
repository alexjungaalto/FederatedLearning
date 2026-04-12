#!/usr/bin/env python3
"""
DiffPriv_ToyExample.py — Differential Privacy Toy Example
----------------------------------------------------------
Fetches max daily temperature (tmax) from all FMI stations for 1 March 2025,
then generates data files and plots for the DiffPriv lecture slides.

Outputs:
  - diffpriv_tmax_01mar2025.csv     cached station data
  - diffpriv_stemplot.csv           station index, name, tmax (sorted by tmax)
  - diffpriv_noisy_stemplot.csv     same with added noise
  - prints LaTeX coordinate strings for pgfplots stem plots

Usage:
    python DiffPriv_ToyExample.py
"""

import os
import requests
import xml.etree.ElementTree as ET
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(SCRIPT_DIR, "diffpriv_tmax_01mar2025.csv")
SIGMA      = 6.0    # noise std dev for the noisy version

# ── FMI WFS parser (reused from Session 3) ───────────────────────────────────

NS = {
    "wfs":    "http://www.opengis.net/wfs/2.0",
    "wml2":   "http://www.opengis.net/waterml/2.0",
    "gml":    "http://www.opengis.net/gml/3.2",
    "target": "http://xml.fmi.fi/namespace/om/atmosphericfeatures/1.1",
    "om":     "http://www.opengis.net/om/2.0",
}

def _parse_fmi_xml(content):
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


# ── Fetch data ───────────────────────────────────────────────────────────────

if os.path.exists(CACHE_FILE):
    print(f"Loading cached data from {CACHE_FILE}")
    df = pd.read_csv(CACHE_FILE)
else:
    print("Fetching tmax for all FMI stations on 2025-03-01 …")
    # Use bounding box for entire Finland
    url = (
        "http://opendata.fmi.fi/wfs?service=WFS&version=2.0.0"
        "&request=getFeature"
        "&storedquery_id=fmi::observations::weather::daily::timevaluepair"
        "&bbox=19.0,59.5,31.6,70.2"
        "&starttime=2025-03-01T00:00:00Z"
        "&endtime=2025-03-01T00:00:00Z"
        "&parameters=tmax&timestep=720"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = _parse_fmi_xml(r.content)
    df.to_csv(CACHE_FILE, index=False)
    print(f"  Saved {len(df)} records to {CACHE_FILE}")

df = df.dropna(subset=["tmax"]).reset_index(drop=True)
print(f"  {len(df)} stations with valid tmax on 2025-03-01\n")


# ── Subsample to ~20 evenly spaced stations ──────────────────────────────────
# From here on we work only with the subsampled set.

df = df.sort_values("tmax", ascending=False).reset_index(drop=True)
step = max(1, len(df) // 20)
df = df.iloc[::step].reset_index(drop=True)
df["idx"] = range(1, len(df) + 1)
n = len(df)

avg = df["tmax"].mean()
print(f"  Subsampled to {n} stations")
print(f"  Average tmax = {avg:.2f} °C")
print(f"  Min = {df['tmax'].min():.1f}, Max = {df['tmax'].max():.1f}\n")

for _, r in df.iterrows():
    print(f"    {int(r['idx']):3d}  {r['station']:40s}  {r['tmax']:+.1f} °C")


# ── Generate noisy version ───────────────────────────────────────────────────

rng = np.random.default_rng(42)
df["tmax_noisy"] = df["tmax"] + rng.normal(0, SIGMA / 2, size=n)


# ── Save CSV ─────────────────────────────────────────────────────────────────

out_csv = os.path.join(SCRIPT_DIR, "diffpriv_stemplot.csv")
df[["idx", "station", "tmax", "tmax_noisy"]].to_csv(out_csv, index=False)
print(f"\n  Saved → {out_csv}")


# ── Print LaTeX pgfplots coordinates ─────────────────────────────────────────

print("\n% --- True values ---")
print("    " + " ".join(
    f"({r['idx']}, {r['tmax']:.1f})" for _, r in df.iterrows()))

print(f"\n% --- Average: y = {avg:.1f} ---")

print("\n% --- Noisy values ---")
print("    " + " ".join(
    f"({r['idx']}, {r['tmax_noisy']:.1f})" for _, r in df.iterrows()))

hi = df.iloc[n // 3]
print(f"\n% --- Highlight: idx={hi['idx']}, {hi['station']}, "
      f"tmax={hi['tmax']:.1f} ---")


# ── Plot ─────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Differential Privacy Toy Example — {n} FMI stations, "
             f"1 March 2025", fontsize=13, fontweight="bold")

# Left: true stem plot
markerline, stemlines, baseline = ax1.stem(
    df["idx"], df["tmax"], linefmt="C0-", markerfmt="C0o", basefmt="k-")
plt.setp(stemlines, linewidth=1.0, alpha=0.7)
plt.setp(markerline, markersize=5)
ax1.axhline(avg, color="green", linestyle="--", linewidth=1.5,
            label=f"$\\bar{{y}} = {avg:.1f}$°C")
ax1.plot(hi["idx"], hi["tmax"], "ro", markersize=9, zorder=5,
         label=f"station {int(hi['idx'])}: {hi['station'].split()[0]}")
ax1.set_xlabel("station index (sorted by tmax)")
ax1.set_ylabel("tmax (°C)")
ax1.set_title("True values — each station is unique")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.25)

# Right: true vs noisy
ax2.stem(df["idx"], df["tmax"], linefmt="C0-", markerfmt="C0o",
         basefmt="k-", label="true $y^{(i)}$")
ml2, sl2, bl2 = ax2.stem(
    df["idx"] + 0.3, df["tmax_noisy"], linefmt="C3-", markerfmt="C3^",
    basefmt="k-", label=f"noisy $y^{{(i)}} + \\varepsilon/2$  (σ={SIGMA}°C)")
plt.setp(sl2, linewidth=0.8, alpha=0.6)
plt.setp(ml2, markersize=5)
ax2.set_xlabel("station index (sorted by true tmax)")
ax2.set_ylabel("observed value (°C)")
ax2.set_title(f"True vs. noisy (σ = {SIGMA}°C) — ordering scrambled")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.25)

fig.tight_layout(rect=[0, 0, 1, 0.94])
out_png = os.path.join(SCRIPT_DIR, "diffpriv_toy_example.png")
fig.savefig(out_png, dpi=120, bbox_inches="tight")
print(f"\n  Saved plot → {out_png}")

plt.show()
print("  Done.")