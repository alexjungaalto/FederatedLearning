#!/usr/bin/env python3
"""
FLNetworks_FMI.py
-----------------
Compares three FL-network constructions over FMI weather stations.
All three connect each station to its K nearest neighbours;
they differ only in the feature vector used to measure "similarity"
between stations.

  Construction 1 — Geographic:
      feature = (lat, lon)
      Two stations are similar if they are close on the map.

  Construction 2 — Climate mean:
      feature = (avg_tmax_Jan, avg_tmax_Aug)
      Two stations are similar if their average summer/winter maximum
      temperatures are close — a simple 2-D climate fingerprint.

  Construction 3 — Climate Gaussian:
      feature = [μ_Jan, vech(Σ_Jan), μ_Aug, vech(Σ_Aug)]
      For each month a 2-D Gaussian N(μ, Σ) is fit to the paired
      (tmin, tmax) daily observations.  μ captures the mean temperature
      level; Σ captures variance and the tmin↔tmax correlation.
      The half-vectorisation vech(Σ) = [σ²_min, σ_min·σ_max·ρ, σ²_max]
      keeps only the 3 unique entries of the symmetric 2×2 matrix.
      Full feature dimension: 2*(2+3) = 10.

Historical months used: January 2026 (winter) and August 2025 (summer).

The key FL insight: the graph topology determines *which local models
are aggregated together*.  Constructions 2 and 3 group stations by
climate similarity rather than geography, so the resulting network can
connect distant stations that share a weather regime while keeping
geographically close but climatically different stations separate.
"""

import os
import csv
import datetime as dt
import requests
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import networkx as nx
from shapely.geometry import Polygon, MultiPolygon
from fmiopendata.wfs import download_stored_query
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

# Each station is connected to its KNN_K nearest neighbours.
# Increasing k makes the graph denser; k=1 gives a minimum spanning tree-like structure.
KNN_K = 4

# Directory that contains this script — used for all relative file paths
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Cache file paths ──────────────────────────────────────────────────────────
# On first run each dataset is fetched from the FMI API and written to CSV.
# On subsequent runs the CSV is loaded directly, skipping the network call.
#
#   fmi_stations.csv      — all active stations (name, lat, lon)
#   fmi_daily_jan.csv     — daily tmin/tmax for January 2026  (long format)
#   fmi_daily_aug.csv     — daily tmin/tmax for August 2025   (long format)
#
# Long format for daily files: one row per (station, parameter, value).
# This handles a variable number of observations per station naturally.

CACHE_STATIONS = os.path.join(DEMO_DIR, "fmi_stations.csv")
CACHE_JAN      = os.path.join(DEMO_DIR, "fmi_daily_jan.csv")
CACHE_AUG      = os.path.join(DEMO_DIR, "fmi_daily_aug.csv")


def save_stations_csv(path: str, names: list, lats: list, lons: list) -> None:
    """Write station list to CSV with columns: station, lat, lon."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["station", "lat", "lon"])
        for name, lat, lon in zip(names, lats, lons):
            w.writerow([name, lat, lon])


def load_stations_csv(path: str):
    """Read station list CSV; returns (names, lats, lons) as lists."""
    names, lats, lons = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.append(row["station"])
            lats.append(float(row["lat"]))
            lons.append(float(row["lon"]))
    return names, lats, lons


def save_daily_csv(path: str, data: dict) -> None:
    """
    Write daily data dict to CSV in long format.

    data : station_name -> {"tmin": [float, ...], "tmax": [float, ...]}
    Columns: station, param, value
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["station", "param", "value"])
        for stn, params in data.items():
            for param, values in params.items():
                for v in values:
                    w.writerow([stn, param, v])


def load_daily_csv(path: str) -> dict:
    """
    Read long-format daily CSV back into a data dict.

    Returns station_name -> {"tmin": [float, ...], "tmax": [float, ...]}
    """
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stn, param, value = row["station"], row["param"], float(row["value"])
            data.setdefault(stn, {}).setdefault(param, []).append(value)
    return data


# ── 1. Finland boundary ──────────────────────────────────────────────────────
# We use the Natural Earth 1:110m countries shapefile (bundled in demos/).
# GeoPandas reads it into a GeoDataFrame; we extract Finland's polygon and
# unpack its exterior ring into (lon, lat) sequences for plotting.

shapefile_path = os.path.join(DEMO_DIR, "ne_110m_admin_0_countries", "ne_110m_admin_0_countries.shp")
world = gpd.read_file(shapefile_path)
finland_row = world[(world["ADMIN"] == "Finland") | (world["SOVEREIGNT"] == "Finland")]
geometry = finland_row["geometry"].iloc[0]

coords_boundary = []
if isinstance(geometry, Polygon):
    # Simple polygon — just one exterior ring
    coords_boundary = list(geometry.exterior.coords)
elif isinstance(geometry, MultiPolygon):
    # Finland has islands → MultiPolygon; collect all exterior rings
    for poly in geometry.geoms:
        coords_boundary.extend(list(poly.exterior.coords))

# Close the polygon ring so matplotlib draws a continuous outline
if coords_boundary[0] != coords_boundary[-1]:
    coords_boundary.append(coords_boundary[0])

boundary_lons, boundary_lats = zip(*coords_boundary)

# ── 2. Live FMI station list → subsample ≈15 % via k-means ─────────────────
# The multipointcoverage query returns all active stations inside the bbox
# that reported at least once in the last 10 hours.  We use this only to
# discover station names and coordinates — not the actual measurements.
#
# Because the full set (~190 stations) would produce a cluttered figure,
# we reduce it to a geographically spread subset using k-means:
#   • partition stations into n_sample clusters in (lat, lon) space
#   • pick the station closest to each cluster centroid as the representative

if os.path.exists(CACHE_STATIONS):
    print(f"Loading station list from cache ({CACHE_STATIONS}) …")
    all_names, all_lats, all_lons = load_stations_csv(CACHE_STATIONS)
    print(f"  {len(all_names)} stations loaded.")
else:
    end_t   = dt.datetime.utcnow()
    start_t = end_t - dt.timedelta(hours=10)
    bbox_live = "19,59.859,32.035,70.170"   # bounding box covering all of Finland

    print("Fetching live FMI station list …")
    obs = download_stored_query(
        "fmi::observations::weather::multipointcoverage",
        args=[
            "bbox=" + bbox_live,
            "starttime=" + start_t.isoformat(timespec="seconds") + "Z",
            "endtime="   + end_t.isoformat(timespec="seconds")   + "Z",
            "timeseries=True",
        ],
    )

    all_lats, all_lons, all_names = [], [], []
    for stn in obs.data.keys():
        m = obs.location_metadata.get(stn, {})
        if m.get("latitude") is not None:
            all_lats.append(m["latitude"])
            all_lons.append(m["longitude"])
            all_names.append(stn)
    print(f"  {len(all_names)} stations found.")

    save_stations_csv(CACHE_STATIONS, all_names, all_lats, all_lons)
    print(f"  Saved to {CACHE_STATIONS}.")

# k-means in (lat, lon) space to get a spread-out subsample
all_latlon = np.array(list(zip(all_lats, all_lons)))
n_sample   = max(10, int(len(all_latlon) * 0.15))
km = KMeans(n_clusters=n_sample, random_state=42, n_init=10)
km.fit(all_latlon)

# For each cluster pick the real station nearest to the centroid
sample_idx = [
    np.where(km.labels_ == k)[0][
        np.argmin(cdist([km.cluster_centers_[k]], all_latlon[km.labels_ == k]))
    ]
    for k in range(n_sample)
]

s_lats  = [all_lats[i]  for i in sample_idx]
s_lons  = [all_lons[i]  for i in sample_idx]
s_names = [all_names[i] for i in sample_idx]
print(f"  Sampled {n_sample} representative stations.")

# ── 3. Historical daily tmin / tmax ─────────────────────────────────────────
# The WFS stored query fmi::observations::weather::daily::timevaluepair
# returns one time-value pair per day per station per parameter.
# The API does not support a 24-hour timestep directly, so intermediate
# records (hour != 0) are skipped; only midnight-stamped daily summaries
# are kept.  Missing or "NaN" values are silently dropped.

NS = {
    "wfs":    "http://www.opengis.net/wfs/2.0",
    "wml2":   "http://www.opengis.net/waterml/2.0",
    "gml":    "http://www.opengis.net/gml/3.2",
    "target": "http://xml.fmi.fi/namespace/om/atmosphericfeatures/1.1",
    "om":     "http://www.opengis.net/om/2.0",
}

def fetch_daily(start_iso: str, end_iso: str, bbox: str = "19.0,59.5,31.6,70.2") -> dict:
    """
    Query the FMI WFS for daily tmin and tmax over a date range.

    Parameters
    ----------
    start_iso, end_iso : ISO-8601 strings, e.g. "2026-01-01T00:00:00Z"
    bbox               : "lon_min,lat_min,lon_max,lat_max"

    Returns
    -------
    dict  station_name -> {"tmin": [float, ...], "tmax": [float, ...]}
        Only days with valid (non-NaN) values are included.
    """
    url = (
        "http://opendata.fmi.fi/wfs?service=WFS&version=2.0.0"
        "&request=getFeature"
        "&storedquery_id=fmi::observations::weather::daily::timevaluepair"
        f"&bbox={bbox}&starttime={start_iso}&endtime={end_iso}"
        "&parameters=tmin,tmax&timestep=720"
    )
    r = requests.get(url, timeout=60)
    assert r.status_code == 200, f"FMI fetch failed: {r.status_code}"

    root = ET.fromstring(r.content)
    data = {}   # station_name -> {param_name -> [float]}

    # Each <wfs:member> element contains data for one station + one parameter
    for member in root.findall(".//wfs:member", NS):
        # --- station name ---
        loc = member.find(".//target:Location", NS)
        if loc is None:
            continue
        name_el = loc.find(
            './gml:name[@codeSpace="http://xml.fmi.fi/namespace/locationcode/name"]', NS
        )
        if name_el is None:
            continue
        stn = name_el.text

        # --- parameter name (tmin or tmax) extracted from the xlink href ---
        param_el = member.find(".//om:observedProperty", NS)
        if param_el is None:
            continue
        href  = param_el.get("{http://www.w3.org/1999/xlink}href", "")
        param = href.split("param=")[1].split("&")[0] if "param=" in href else ""

        # --- daily values: keep only midnight-stamped records ---
        values = []
        for pt in member.findall(".//wml2:point", NS):
            t_el = pt.find(".//wml2:time", NS)
            v_el = pt.find(".//wml2:value", NS)
            if t_el is None or v_el is None:
                continue
            t = dt.datetime.strptime(t_el.text, "%Y-%m-%dT%H:%M:%SZ")
            if t.hour != 0:
                continue   # skip intra-day duplicates
            try:
                values.append(float(v_el.text))
            except (TypeError, ValueError):
                pass   # "NaN" string or missing → skip

        if stn not in data:
            data[stn] = {}
        data[stn].setdefault(param, []).extend(values)

    return data


if os.path.exists(CACHE_JAN):
    print(f"Loading January 2026 data from cache ({CACHE_JAN}) …")
    data_jan = load_daily_csv(CACHE_JAN)
    print(f"  {len(data_jan)} stations loaded.")
else:
    print("Fetching daily tmin/tmax for January 2026 …")
    data_jan = fetch_daily("2026-01-01T00:00:00Z", "2026-01-31T00:00:00Z")
    print(f"  {len(data_jan)} stations returned.")
    save_daily_csv(CACHE_JAN, data_jan)
    print(f"  Saved to {CACHE_JAN}.")

if os.path.exists(CACHE_AUG):
    print(f"Loading August 2025 data from cache ({CACHE_AUG}) …")
    data_aug = load_daily_csv(CACHE_AUG)
    print(f"  {len(data_aug)} stations loaded.")
else:
    print("Fetching daily tmin/tmax for August 2025 …")
    data_aug = fetch_daily("2025-08-01T00:00:00Z", "2025-08-31T00:00:00Z")
    print(f"  {len(data_aug)} stations returned.")
    save_daily_csv(CACHE_AUG, data_aug)
    print(f"  Saved to {CACHE_AUG}.")

# ── 4. Per-station feature vectors ──────────────────────────────────────────

def month_stats(data: dict, stn: str):
    """
    Extract paired (tmin, tmax) arrays for a station from a monthly data dict.

    Returns (tmin_arr, tmax_arr) keeping only days where both values are
    finite.  Returns (None, None) if fewer than 5 valid days are available
    (too few to estimate a covariance reliably).
    """
    rec  = data.get(stn, {})
    tmin = np.array(rec.get("tmin", []), dtype=float)
    tmax = np.array(rec.get("tmax", []), dtype=float)
    ok   = np.isfinite(tmin) & np.isfinite(tmax)   # valid-day mask
    if ok.sum() < 5:
        return None, None
    return tmin[ok], tmax[ok]


def gauss_vec(tmin_arr, tmax_arr):
    """
    Fit a 2-D Gaussian to (tmin, tmax) observations and return a flat
    parameter vector.

    The distribution is N(μ, Σ) where
        μ  = [E[tmin], E[tmax]]           (2-D mean)
        Σ  = [[var(tmin),   cov],          (2×2 sample covariance)
               [cov,   var(tmax)]]

    We store the half-vectorisation vech(Σ) = [Σ₀₀, Σ₀₁, Σ₁₁] to avoid
    redundancy (Σ is symmetric), giving a 5-D vector per month:
        [μ_tmin, μ_tmax, var_tmin, cov_tmin_tmax, var_tmax]
    """
    X    = np.column_stack([tmin_arr, tmax_arr])    # shape (n_days, 2)
    mu   = X.mean(axis=0)                           # (2,)
    cov  = np.cov(X.T)                              # (2,2) sample covariance
    vech = np.array([cov[0, 0], cov[0, 1], cov[1, 1]])  # upper triangle
    return np.concatenate([mu, vech])               # (5,)


# Iterate over the sampled stations and build feature rows.
# Stations missing data in either month are excluded from G2 and G3
# (climate_idx tracks which sampled stations pass this filter).
feat2_rows  = []   # rows of shape (2,)  for Construction 2
feat3_rows  = []   # rows of shape (10,) for Construction 3
climate_idx = []   # indices into s_names with complete climate data

for i, stn in enumerate(s_names):
    tmin_j, tmax_j = month_stats(data_jan, stn)
    tmin_a, tmax_a = month_stats(data_aug, stn)
    if tmin_j is None or tmin_a is None:
        continue   # insufficient data — skip for climate-based graphs

    # Construction 2: scalar average tmax for each month → 2-D feature
    feat2_rows.append([tmax_j.mean(), tmax_a.mean()])

    # Construction 3: concatenate Gaussian params for both months → 10-D feature
    feat3_rows.append(np.concatenate([
        gauss_vec(tmin_j, tmax_j),   # 5-D winter Gaussian
        gauss_vec(tmin_a, tmax_a),   # 5-D summer Gaussian
    ]))
    climate_idx.append(i)

feat2 = np.array(feat2_rows)   # (m, 2)
feat3 = np.array(feat3_rows)   # (m, 10)
print(f"  {len(climate_idx)} / {n_sample} stations have sufficient climate data.")

# ── 5. KNN graph construction ────────────────────────────────────────────────
# All three graphs use the same rule: connect each node to its k nearest
# neighbours in feature space.  Features are z-score normalised first so
# that dimensions with different scales (e.g. degrees vs. °C) contribute
# equally to the Euclidean distance.
#
# Note: NearestNeighbors returns k+1 neighbours (including the point itself
# at distance 0), so we skip index 0 (nbrs[1:]).  The resulting graph is
# undirected — if A→B and B→A both appear, networkx deduplicates the edge.

def knn_graph(names, features, k):
    """
    Build an undirected NetworkX graph by connecting each node to its
    k nearest neighbours in (normalised) feature space.

    Parameters
    ----------
    names    : list of str, node identifiers
    features : (n, d) array, one row per node
    k        : number of neighbours

    Returns
    -------
    nx.Graph with edge attribute 'weight' = normalised Euclidean distance
    """
    X  = StandardScaler().fit_transform(features)   # z-score normalise
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    distances, indices = nn.kneighbors(X)            # shape (n, k+1)

    G = nx.Graph()
    G.add_nodes_from(names)
    for i, nbrs in enumerate(indices):
        for j, dist in zip(nbrs[1:], distances[i, 1:]):   # skip self (index 0)
            G.add_edge(names[i], names[j], weight=round(float(dist), 4))
    return G


# Construction 1: geographic — all sampled stations have lat/lon
latlon_arr = np.array(list(zip(s_lats, s_lons)))
G1 = knn_graph(s_names, latlon_arr, KNN_K)

# Constructions 2 & 3: climate-based — restricted to stations with data
climate_names = [s_names[i] for i in climate_idx]
climate_lats  = [s_lats[i]  for i in climate_idx]
climate_lons  = [s_lons[i]  for i in climate_idx]
G2 = knn_graph(climate_names, feat2, KNN_K)
G3 = knn_graph(climate_names, feat3, KNN_K)

# Store geographic coordinates as node attributes for plotting
# (the graph itself is built in feature space, not geographic space)
for name, lat, lon in zip(climate_names, climate_lats, climate_lons):
    G2.nodes[name]["lat"] = lat;  G2.nodes[name]["lon"] = lon
    G3.nodes[name]["lat"] = lat;  G3.nodes[name]["lon"] = lon
for name, lat, lon in zip(s_names, s_lats, s_lons):
    G1.nodes[name]["lat"] = lat;  G1.nodes[name]["lon"] = lon

print(f"G1 (lat/lon):           {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges")
print(f"G2 (mean tmax):         {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges")
print(f"G3 (Gaussian tmin/max): {G3.number_of_nodes()} nodes, {G3.number_of_edges()} edges")

# ── 6. Figure ────────────────────────────────────────────────────────────────
# All three subplots share the same geographic canvas (Finland boundary +
# station dots placed at their true lat/lon).  Only the edges differ.
# Long-range edges in G2/G3 indicate stations that are geographically
# distant but climatically similar.

titles = [
    f"(1) Geographic\nfeature: (lat, lon)  |  k={KNN_K}",
    f"(2) Climate mean\nfeature: (avg tmax Jan, avg tmax Aug)  |  k={KNN_K}",
    f"(3) Climate Gaussian\nfeature: μ, Σ of (tmin, tmax) Jan+Aug  |  k={KNN_K}",
]
graphs = [G1, G2, G3]

fig, axes = plt.subplots(1, 3, figsize=(15, 9))

for ax, G, title in zip(axes, graphs, titles):
    # Node positions are always geographic (lon, lat) regardless of which
    # feature space was used to determine the edges
    pos = {n: (G.nodes[n]["lon"], G.nodes[n]["lat"]) for n in G.nodes}

    ax.plot(boundary_lons, boundary_lats, color="black", linewidth=1.0, zorder=1)
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color="steelblue", alpha=0.5, width=1.2)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color="blue", node_size=50,
                           edgecolors="white", linewidths=0.5)
    ax.set_title(title, fontsize=9, pad=6)
    ax.axis("off")

fig.suptitle(
    "FL Network over FMI Stations — three edge constructions (KNN)",
    fontsize=12, y=1.01,
)
fig.tight_layout()

out_path = "FLNetworks_FMI.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Figure saved to {out_path}")
plt.show()