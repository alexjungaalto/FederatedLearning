#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 18:35:24 2025

@author: junga1
"""

from fmiopendata.wfs import download_stored_query
import datetime as dt
import matplotlib.pyplot as plt
import random
import geopandas as gpd
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, MultiPolygon
import csv
import networkx as nx
import copy
from numpy import linalg as LA 

def connect_nearest_neighbors(graph, min_neighbors=4):
    """
    Connect each node in the NetworkX graph to its minimum number of nearest neighbors
    based on the Euclidean distance of the "z" attribute.

    Args:
        graph (nx.Graph): A NetworkX graph where each node has a "z" attribute as a NumPy array.
        min_neighbors (int): Minimum number of neighbors for each node.
    """
    graph.remove_edges_from(list(graph.edges))
    # Extract node indices and their "z" attributes
    nodes = list(graph.nodes(data=True))
    node_indices = [node[0] for node in nodes]
    z_values = np.array([data["z"] for _, data in nodes])
    print(z_values.shape)
    # Compute pairwise distances
    distance_matrix = cdist(z_values, z_values, metric="euclidean")

    # Connect each node to its minimum number of nearest neighbors
    for i, node_a in enumerate(node_indices):
        # Get indices of the smallest distances (excluding self-loop)
        nearest_indices = np.argsort(distance_matrix[i])[1:min_neighbors+1]
        for j in nearest_indices:
            node_b = node_indices[j]
            dist = distance_matrix[i, j]

            # Add edge with weight if it doesn't already exist
            if not graph.has_edge(node_a, node_b):
                graph.add_edge(node_a, node_b, weight=dist)


def plotFMI(G,lons,lats,fname="test",title=""): 
    """Generates a scatter plot of FMI stations using latitude and longitude as coordinates.

    Parameters
    ----------
    G : networkx.Graph
        A graph where each node represents an FMI station, and each node has a 'coord' attribute 
        containing a tuple (latitude, longitude).

    Returns
    -------
    None
        Displays a scatter plot with nodes (stations) plotted based on their coordinates. Nodes 
        are labeled, and edges between stations are drawn as lines.
    """
    # Extract coordinates
    coords = np.array([G.nodes[node]['coord'] for node in G.nodes])

    # Create the plot
    fig, ax = plt.subplots()
    
    ax.plot(lons, lats, marker='o', linestyle='-', markersize=5)
    # Draw nodes
    ax.scatter(coords[:, 1], coords[:, 0], color='black', s=40, zorder=5)
    
    # Add labels
    for node, (lat, lon) in enumerate(coords):
        ax.text(lon + 0.1, lat + 0.2, str(node+1), fontsize=20, color='black')

    # Draw edges
    for u, v in G.edges:
        ax.plot([coords[u, 1], coords[v, 1]], [coords[u, 0], coords[v, 0]], linestyle='-', color='gray')

    # Set labels and title
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_title(title)

    # Save the figure as a PNG file
    plt.savefig(fname+".png", dpi=100, bbox_inches='tight')

    # Show the plot (optional)
    plt.show()


#############################
# Step 1: Construct a polygon resembling Finland's shape
#############################

# Path to the shapefile containing country boundaries
shapefile_path = "ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"

# Read the shapefile into a GeoDataFrame
world = gpd.read_file(shapefile_path)

# Filter rows corresponding to Finland
finland_rows = world[(world['ADMIN'] == 'Finland') | (world['SOVEREIGNT'] == 'Finland')]

# Extract geometry for Finland (POLYGON or MULTIPOLYGON)
geometry = finland_rows['geometry'].iloc[0]

# Initialize a list to store coordinates of the polygon
coordinates = []

# Check the geometry type and extract exterior coordinates
if isinstance(geometry, Polygon):
    # Single Polygon: Get exterior coordinates
    coordinates = list(geometry.exterior.coords)
elif isinstance(geometry, MultiPolygon):
    # MultiPolygon: Extract coordinates from each sub-polygon
    for polygon in geometry:
        coordinates.extend(list(polygon.exterior.coords))

# Ensure the polygon is closed by appending the first point to the end
if coordinates[0] != coordinates[-1]:
    coordinates.append(coordinates[0])

# Separate longitude and latitude for visualization
lons, lats = zip(*coordinates)

#############################
# Step 2: Retrieve weather data from FMI
#############################

# Define the time range for the data
end_time = dt.datetime.utcnow()  # Current UTC time

end_time = dt.datetime.strptime("2024-05-15T16:10:08Z", "%Y-%m-%dT%H:%M:%SZ")

start_time = end_time - dt.timedelta(hours=10)  # One hour earlier

# Format time strings for the FMI query
start_time = start_time.isoformat(timespec="seconds") + "Z"
end_time = end_time.isoformat(timespec="seconds") + "Z"

# the code snippet below reads in data from web interface 

bbox="19,59.859,32.035,70.170"  #   for entire finlan 

#bbox = 21.02, 60.7, 21.03, 60.73 for Kustavi Isokaari


obs = download_stored_query("fmi::observations::weather::multipointcoverage",
                            args=["bbox="+bbox,
                                  "starttime=" + start_time,
                                  "endtime=" + end_time,
                                  "timeseries=True"])

# Initialize lists to store station metadata
latitudes = []
longitudes = []
fmisids = []
stationname = []

# Extract metadata for each station
for station in obs.data.keys():
    metadata = obs.location_metadata.get(station, {})
    lat = metadata.get("latitude")
    lon = metadata.get("longitude")
    fmisid = metadata.get("fmisid")

    if lat is not None and lon is not None:
        latitudes.append(lat)
        longitudes.append(lon)
        fmisids.append(fmisid)
        stationname.append(station)

# Combine latitude and longitude for clustering
coordinates = np.array(list(zip(latitudes, longitudes)))

#############################
# Step 3: Sample stations using K-Means clustering
#############################

# Determine the sample size (e.g., 10% of all stations)
sample_size = int(len(coordinates) * 0.08)

# Apply K-Means clustering to group nearby stations
kmeans = KMeans(n_clusters=sample_size, random_state=42)
kmeans.fit(coordinates)

# Identify the station closest to each cluster center
sample_indices = []
for cluster_label in range(sample_size):
    # Indices of points in the current cluster
    cluster_points = np.where(kmeans.labels_ == cluster_label)[0]

    # Calculate distances to the cluster center
    cluster_center = kmeans.cluster_centers_[cluster_label]
    distances = cdist([cluster_center], coordinates[cluster_points])

    # Find the closest station to the center
    closest_index_within_cluster = cluster_points[np.argmin(distances)]
    sample_indices.append(closest_index_within_cluster)

# Extract metadata for the sampled stations
sample_longitudes = [longitudes[i] for i in sample_indices]
sample_latitudes = [latitudes[i] for i in sample_indices]
sample_stationname = [stationname[i] for i in sample_indices]

#############################
# Step 4: Save sampled station data to a CSV file
#############################

# Combine sampled station data into dictionaries
stations_data = [
    {"Station": station, "lat": lat, "lon": lon}
    for station, lat, lon in zip(sample_stationname, sample_latitudes, sample_longitudes)
]

# Filepath for the output CSV
output_csv_file = "fmi_stations_subset.csv"

# Write station data to the CSV file
with open(output_csv_file, mode="w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["Station", "lat", "lon"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(stations_data)

print(f"Data successfully written to {output_csv_file}")

#############################
# Step 5: Visualize the data
#############################

# Plot the sampled stations and Finland's polygon
plt.figure(figsize=(10, 8))
plt.scatter(sample_longitudes, sample_latitudes, c='blue', alpha=1, s=100, edgecolors='k')
plt.plot(lons, lats, marker='o', linestyle='-', markersize=5)

# Annotate the sampled stations with names
for i, name in enumerate(sample_stationname):
    plt.text(sample_longitudes[i], sample_latitudes[i] + 0.2, name.split()[0], fontsize=20, ha='right')

# Customize plot
plt.axis('off')
plt.legend()
plt.savefig("FMIStations.png", dpi=50, bbox_inches='tight')
plt.show()

# Plot only sampled stations 
plt.figure(figsize=(10, 8))
plt.scatter(sample_longitudes, sample_latitudes, c='blue', alpha=1, s=100, edgecolors='k')
plt.plot(lons, lats, marker='o', linestyle='-', markersize=5)

# Annotate the sampled stations with names
for i, name in enumerate(sample_stationname):
    plt.text(sample_longitudes[i], sample_latitudes[i] + 0.2, str(i+1), fontsize=40, ha='right')

# Customize plot
plt.axis('off')
plt.legend()
plt.savefig("FMINetwork.png", dpi=50, bbox_inches='tight')
plt.show()

#############################
# Step 6: Create a network graph with station data
#############################

# Initialize a graph
G = nx.Graph()


# Add a one node per station
G.add_nodes_from(range(sample_size))



# Add nodes to the graph for each sampled station
for i, station in enumerate(sample_stationname):
    # Extract weather data for the station
    data_dict = obs.data[station]

    # Create a DataFrame for the weather data
    df = pd.DataFrame({
        'Time': data_dict['times'],
        'Air Temperature (degC)': data_dict['Air temperature']['values'],
        'Wind Speed (m/s)': data_dict['Wind speed']['values'],
        'Gust Speed (m/s)': data_dict['Gust speed']['values'],
        'Wind Direction (deg)': data_dict['Wind direction']['values'],
        'Relative Humidity (%)': data_dict['Relative humidity']['values'],
        'Dew-Point Temperature (degC)': data_dict['Dew-point temperature']['values'],
        'Precipitation Amount (mm)': data_dict['Precipitation amount']['values'],
        'Pressure (msl) (hPa)': data_dict['Pressure (msl)']['values']
    })

    # Add a node with the station name and its DataFrame as attributes
    G.add_node(i, stationname=station,y=df["Air Temperature (degC)"].values,coord=(sample_latitudes[i], sample_longitudes[i]))



# Example of accessing the attributes of each node in the graph
for node_id, attributes in G.nodes(data=True):
    station_name = attributes["stationname"]
    temperature_data = attributes["y"]  # Access the temperature data
    coordinates = attributes["coord"]  # Access the coordinates
    
    # Example output (can be replaced with actual processing logic)
    print(f"Node {node_id}:")
    print(f"  Station Name: {station_name}")
    print(f"  Temperature Data: {temperature_data}")
    print(f"  Coordinates: {coordinates}")

for node in G.nodes:
    G.nodes[node]['z'] = np.array(G.nodes[node]['coord']) # Replace 42 with the desired value

connect_nearest_neighbors(G, 3)

plotFMI(G, lons, lats,"coords","use latitude and longitude for placing edges")

# Compute the average of the "y" attribute for each node and store it in "z"
for node in G.nodes:
    y_tuple = G.nodes[node]['y']
    y_array = np.array(y_tuple)
    G.nodes[node]['z'] = np.mean(y_array).reshape(1,)
    
connect_nearest_neighbors(G, 3)

plotFMI(G, lons, lats,"avgtemp","use avg temperature for placing edges")
    
    
    
# # Example of accessing the DataFrame for a specific node
# for node in G.nodes(data=True):
#     station_name = node["stationname"]
#     station_data = node[1]['y']
#     print(f"Station: {station_name}")
#     #print(station_data[["Time","Air Temperature (degC)"]].head())

print("Graph construction complete with weather data for each station.")
