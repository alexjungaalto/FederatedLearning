
"""
CS-E4740 Federated Learning — Assignment A1 (Starter Code)

============================================================
A1: From ML to FL using FMI daily temperature data
============================================================

This file IS the assignment.

You do NOT submit anything.
You run this file locally and answer a MyCourses quiz
based on the printed outputs.

------------------------------------------------------------
CSV FORMAT (provided via MyCourses)
------------------------------------------------------------
Each row corresponds to ONE day at ONE FMI station.

Columns:
    station  : station name (string)
    lat      : latitude (float)
    lon      : longitude (float)
    day      : date in YYYY-MM-DD format
    tmax     : daily maximum temperature
    tmin     : daily minimum temperature

------------------------------------------------------------
LEARNING OBJECTIVE
------------------------------------------------------------
Understand how:

    ONE ML model
becomes
    MANY local models + a graph (FL)

by implementing the *simplest possible* FL pipeline.

------------------------------------------------------------
IMPORTANT SIMPLIFICATION
------------------------------------------------------------
• Distance between stations is NOT geodesic.
• We use a simple Euclidean distance on (lat, lon).
• The graph threshold is HARD-CODED.
• Do NOT change constants in this file.

------------------------------------------------------------
"""

# ============================================================
# Imports
# ============================================================

import argparse          # For parsing command-line arguments
import csv               # For reading CSV files
import math              # For sqrt() and exp()
from datetime import datetime  # For parsing dates
import matplotlib.pyplot as plt   # For plotting the FL graph


# ============================================================
# HARD-CODED GRAPH THRESHOLD (DO NOT CHANGE)
# ============================================================

DIST_THRESHOLD = 1.0
# ^ Two stations are connected if their Euclidean distance
#   in (lat, lon) coordinates is <= 1.0.
#   This is a *proxy distance*, not kilometers.
#   The MyCourses quiz assumes this exact value.


# ============================================================
# Distance function (VERY SIMPLE ON PURPOSE)
# ============================================================

def proxy_distance(lat1, lon1, lat2, lon2):
    # Compute difference in latitude
    d_lat = lat1 - lat2

    # Compute difference in longitude
    d_lon = lon1 - lon2

    # Euclidean norm in R^2
    return math.sqrt(d_lat ** 2 + d_lon ** 2)


# ============================================================
# Linear regression: y = w0 + w1 * x
# ============================================================

def fit_linear_regression(xs, ys):
    """
    Closed-form least squares for 1D linear regression.

    Input:
        xs : list of tmin values
        ys : list of tmax values

    Output:
        w0 : intercept
        w1 : slope
    """

    # Safety check: cannot fit with zero samples
    if len(xs) == 0:
        raise ValueError("No samples provided for regression")

    # Compute mean of x values
    x_mean = sum(xs) / len(xs)

    # Compute mean of y values
    y_mean = sum(ys) / len(ys)

    # Initialize numerator of slope formula
    num = 0.0

    # Initialize denominator of slope formula
    den = 0.0

    # Loop over samples to compute covariance and variance
    for x, y in zip(xs, ys):
        num += (x - x_mean) * (y - y_mean)
        den += (x - x_mean) ** 2

    # Handle degenerate case: all x values identical
    if den < 1e-12:
        # Best constant predictor is the mean of y
        return y_mean, 0.0

    # Compute slope
    w1 = num / den

    # Compute intercept
    w0 = y_mean - w1 * x_mean

    return w0, w1


def mean_squared_error(w0, w1, xs, ys):
    """
    Compute mean squared error for linear model.
    """

    # Initialize error accumulator
    err = 0.0

    # Loop over samples
    for x, y in zip(xs, ys):
        # Model prediction
        y_hat = w0 + w1 * x

        # Squared error
        err += (y - y_hat) ** 2

    # Return average squared error
    return err / len(xs)


# ============================================================
# CSV loading
# ============================================================

def load_weather_csv(path):
    """
    Load FMI weather data from CSV.

    Returns:
        station_info:
            dict mapping
                station_name -> (lat, lon)

        station_records:
            dict mapping
                station_name -> list of (day, tmin, tmax)
    """

    # Dictionary for station coordinates
    station_info = {}

    # Dictionary for per-station time series
    station_records = {}

    # Open CSV file
    with open(path, "r", encoding="utf-8") as f:
        # Create CSV reader that uses header row
        reader = csv.DictReader(f)

        # Required column names
        required = {"station", "lat", "lon", "day", "tmax", "tmin"}

        # Check that CSV contains required columns
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV file missing required columns")

        # Iterate over rows
        for row in reader:
            # Read station name
            station = row["station"].strip()

            # Parse coordinates
            lat = float(row["lat"])
            lon = float(row["lon"])

            # Parse date
            day = datetime.strptime(row["day"], "%Y-%m-%d")

            # Parse temperatures
            tmax = float(row["tmax"])
            tmin = float(row["tmin"])

            # Store station coordinates if first time seen
            if station not in station_info:
                station_info[station] = (lat, lon)
                station_records[station] = []

            # Append daily record
            station_records[station].append((day, tmin, tmax))

    # Sort records for each station by date
    for s in station_records:
        station_records[s].sort(key=lambda r: r[0])

    return station_info, station_records


# ============================================================
# Train / validation split
# ============================================================

def split_train_val(records):
    """
    Deterministic split:

        • last day  -> validation
        • all earlier days -> training

    This keeps the assignment simple and reproducible.
    """

    # If only one record exists, use it for both
    if len(records) < 2:
        return records, records

    # Otherwise split normally
    return records[:-1], records[-1:]


# ============================================================
# Build the FL graph
# ============================================================

def build_graph(station_info):
    """
    Construct a weighted undirected graph A.

    Nodes:
        station names

    Edge rule:
        connect i--j if proxy_distance(i,j) <= DIST_THRESHOLD

    Edge weight:
        A_ij = exp(-distance)
    """

    # List of station names
    stations = list(station_info.keys())

    # Initialize empty adjacency dictionary
    A = {s: {} for s in stations}

    # Loop over unordered station pairs
    for i, si in enumerate(stations):
        lat_i, lon_i = station_info[si]

        for sj in stations[i + 1:]:
            lat_j, lon_j = station_info[sj]

            # Compute proxy distance
            d = proxy_distance(lat_i, lon_i, lat_j, lon_j)

            # Add edge if below threshold
            if d <= DIST_THRESHOLD:
                # Weight decays with distance
                w = math.exp(-d)

                # Store symmetrically
                A[si][sj] = w
                A[sj][si] = w

    return A


# ============================================================
# One FL collaboration step
# ============================================================

def one_collaboration_step(A, local_params):
    """
    Perform ONE federated collaboration step.

    For each station i:
        • receive neighbors' parameters
        • replace own parameters by weighted average

    No retraining is done here.
    """

    # Dictionary for updated parameters
    new_params = {}

    # Loop over all stations
    for i in local_params:
        neighbors = A[i]

        # If station is isolated, keep own model
        if len(neighbors) == 0:
            new_params[i] = local_params[i]
            continue

        # Sum of edge weights
        weight_sum = sum(neighbors.values())

        # Initialize weighted sums
        w0 = 0.0
        w1 = 0.0

        # Weighted averaging
        for j, aij in neighbors.items():
            w0 += aij * local_params[j][0]
            w1 += aij * local_params[j][1]

        # Normalize
        new_params[i] = (w0 / weight_sum, w1 / weight_sum)

    return new_params


# ============================================================
# Plot the FL graph (for visualization only)
# ============================================================

def plot_graph(station_info, A):
    """
    Plot the FL graph using (lat, lon) as 2D coordinates.

    Nodes:
        - plotted as points at (lon, lat)

    Edges:
        - drawn as straight lines between connected stations

    IMPORTANT:
        - This plot is only for intuition.
        - It has NO effect on grading or quiz answers.
    """
    # Create a new figure
    plt.figure(figsize=(7, 7))
    # --------------------------------------------------------
    # Plot edges
    # --------------------------------------------------------
    for i in A:
        lat_i, lon_i = station_info[i]

        for j in A[i]:
            lat_j, lon_j = station_info[j]

            # Draw a line between station i and j
            plt.plot(
                [lon_i, lon_j],   # x-coordinates (longitude)
                [lat_i, lat_j],   # y-coordinates (latitude)
                color="gray",
                linewidth=1,
                alpha=0.7
            )
    # --------------------------------------------------------
    # Plot nodes
    # --------------------------------------------------------
    for station, (lat, lon) in station_info.items():
        # Plot station as a blue dot
        plt.scatter(
            lon, lat,
            color="blue",
            s=40,
            zorder=3
        )

        # Annotate station name (small font)
        plt.text(
            lon, lat,
            station,
            fontsize=7,
            ha="left",
            va="bottom"
        )
    # --------------------------------------------------------
    # Plot formatting
    # --------------------------------------------------------
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("FL Graph (Stations connected if distance ≤ threshold)")
    plt.grid(True)
    # Make axes equal so distances look reasonable
    plt.axis("equal")
    # Show the plot
    plt.show()


# ============================================================
# Main pipeline
# ============================================================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True,
                        help="Path to FMI weather CSV file")
    args = parser.parse_args()

    # --------------------------------------------------------
    # (1) Load data
    # --------------------------------------------------------
    station_info, station_records = load_weather_csv(args.csv)

    # Sorted list of station names
    stations = sorted(station_records.keys())

    # --------------------------------------------------------
    # (2) Build FL graph
    # --------------------------------------------------------
    A = build_graph(station_info)
    
    # Visualize the graph (purely illustrative)
    plot_graph(station_info, A)

    # Count nodes
    n_nodes = len(stations)

    # Count undirected edges
    n_edges = sum(len(A[s]) for s in A) // 2

    # --------------------------------------------------------
    # (3) Train local models + build global dataset
    # --------------------------------------------------------
    local_params = {}
    local_splits = {}

    global_x_tr, global_y_tr = [], []
    global_x_va, global_y_va = [], []

    for s in stations:
        # Split into train and validation
        train, val = split_train_val(station_records[s])
        local_splits[s] = (train, val)

        # Extract features and labels
        x_tr = [r[1] for r in train]
        y_tr = [r[2] for r in train]
        x_va = [r[1] for r in val]
        y_va = [r[2] for r in val]

        # Train local model
        w0, w1 = fit_linear_regression(x_tr, y_tr)
        local_params[s] = (w0, w1)

        # Add to global dataset
        global_x_tr.extend(x_tr)
        global_y_tr.extend(y_tr)
        global_x_va.extend(x_va)
        global_y_va.extend(y_va)

    # --------------------------------------------------------
    # (4) Train global model
    # --------------------------------------------------------
    gw0, gw1 = fit_linear_regression(global_x_tr, global_y_tr)

    global_train_mse = mean_squared_error(
        gw0, gw1, global_x_tr, global_y_tr
    )

    global_val_mse = mean_squared_error(
        gw0, gw1, global_x_va, global_y_va
    )

    # --------------------------------------------------------
    # (5) Local validation error BEFORE FL
    # --------------------------------------------------------
    before = []

    for s in stations:
        val = local_splits[s][1]
        xs = [r[1] for r in val]
        ys = [r[2] for r in val]

        w0, w1 = local_params[s]
        before.append(mean_squared_error(w0, w1, xs, ys))

    mean_before = sum(before) / len(before)

    # --------------------------------------------------------
    # (6) ONE FL collaboration step
    # --------------------------------------------------------
    collab_params = one_collaboration_step(A, local_params)

    after = []

    for s in stations:
        val = local_splits[s][1]
        xs = [r[1] for r in val]
        ys = [r[2] for r in val]

        w0, w1 = collab_params[s]
        after.append(mean_squared_error(w0, w1, xs, ys))

    mean_after = sum(after) / len(after)

    # --------------------------------------------------------
    # (7) Print results for MyCourses quiz
    # --------------------------------------------------------
    print("\n=== A1 RESULTS (use these in the MyCourses quiz) ===\n")

    print("[Global model]")
    print(f"train_MSE = {global_train_mse:.6f}")
    print(f"val_MSE   = {global_val_mse:.6f}")

    print("\n[Graph]")
    print(f"DIST_THRESHOLD = {DIST_THRESHOLD}")
    print(f"nodes = {n_nodes}")
    print(f"edges = {n_edges}")

    print("\n[Local models]")
    print(f"mean_val_MSE_before = {mean_before:.6f}")

    print("\n[After ONE FL collaboration step]")
    print(f"mean_val_MSE_after  = {mean_after:.6f}")

    print("\nDone. Answer the MyCourses quiz based on these numbers.")


# Standard Python entry point
if __name__ == "__main__":
    main()
