#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:24:59 2024

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
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

################
#### first we construct a polygon that resembles the shape of Finland 
####################
# Update the path to the location where you've saved the shapefile
shapefile_path = "ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
# Read the shapefile
world = gpd.read_file(shapefile_path)
# grab all African countries
finland_rows = world[(world['ADMIN'] == 'Finland') | (world['SOVEREIGNT'] == 'Finland')]
# Example: Assuming the geometry is stored in a variable `geometry` (POLYGON or MULTIPOLYGON)
geometry = finland_rows['geometry'].iloc[0]  # Replace with your specific row/column
# Initialize a list to store latitude/longitude pairs
coordinates = []

# Check the geometry type and extract coordinates
if isinstance(geometry, Polygon):
    # Extract exterior coordinates for a single Polygon
    coordinates = list(geometry.exterior.coords)
elif isinstance(geometry, MultiPolygon):
    # Extract exterior coordinates for all Polygons in the MultiPolygon
    for polygon in geometry:
        coordinates.extend(list(polygon.exterior.coords))

 # Ensure the polygon closes by appending the first point to the end
if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])

# Extract longitude and latitude
lons, lats = zip(*coordinates)


#################
#### next we read in weather measurements collected by FMI stations 
#### during a specific time period and within a specific geopgraphic area (bbox) 
######################

# Retrieve the latest hour of data from a bounding box
end_time = dt.datetime.utcnow()
start_time = end_time - dt.timedelta(hours=1)
# Convert times to properly formatted strings
start_time = start_time.isoformat(timespec="seconds") + "Z"
# -> 2020-07-07T12:00:00Z
end_time = end_time.isoformat(timespec="seconds") + "Z"
# -> 2020-07-07T13:00:00Z


# the code snippet below reads in data from web interface 

bbox="19,59.859,32.035,70.170"  #   for entire finlan 

#bbox = 21.02, 60.7, 21.03, 60.73 for Kustavi Isokaari


obs = download_stored_query("fmi::observations::weather::multipointcoverage",
                            args=["bbox="+bbox,
                                  "starttime=" + start_time,
                                  "endtime=" + end_time,
                                  "timeseries=True"])

# Extract latitude, longitude, and fmisid
latitudes = []
longitudes = []
fmisids = []
stationname=[]


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


# Combine latitudes and longitudes into a single array for clustering
coordinates = np.array(list(zip(latitudes, longitudes)))


### instead of plotting all stations (which makes the plot crowded) 
### we plot only a sample of satations 

# Define the number of clusters based on the sample size (20% of stations)

sample_size = int(len(coordinates) * 0.1)

# the sample of stations is determined by k-means clustering the lat/lon pairs 
# of all stations. Then we pick those FMI stations that are closest to the cluster means.

kmeans = KMeans(n_clusters=sample_size, random_state=42)
kmeans.fit(coordinates)

# Find the station closest to the cluster mean for each cluster

sample_indices = []
for cluster_label in range(sample_size):
    # Get the indices of all points in the current cluster
    cluster_points = np.where(kmeans.labels_ == cluster_label)[0]
    
    # Calculate distances of all cluster points to the cluster centroid
    cluster_center = kmeans.cluster_centers_[cluster_label]
    distances = cdist([cluster_center], coordinates[cluster_points])
    
    # Find the index of the closest point
    closest_index_within_cluster = cluster_points[np.argmin(distances)]
    sample_indices.append(closest_index_within_cluster)

# Subset the data
sample_longitudes = [longitudes[i] for i in sample_indices]
sample_latitudes = [latitudes[i] for i in sample_indices]
sample_stationname = [stationname[i] for i in sample_indices]

    
for station in obs.data.keys(): 
    print(station + str(obs.location_metadata[station]))

formatted_dates = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in obs.data['Kustavi Isokari']['times']]


plt.figure(figsize=(10, 8))
    

    #plt.plot(polygon_longitudes, polygon_latitudes, c='red', linewidth=2, label="Polygon Approximation of Finland")
plt.scatter(sample_longitudes, sample_latitudes, c='blue', alpha=1, s=100, edgecolors='k')
plt.plot(lons, lats, marker='o', linestyle='-', markersize=5, label='Polygon Border')

    # Annotate the stations with their names
for i, name in enumerate(sample_stationname):
    plt.text(sample_longitudes[i], sample_latitudes[i]+0.2, name.split()[0], fontsize=15, ha='right')
        
    # Remove axes
plt.axis('off')
plt.savefig("FMIStations.png", dpi=100)
plt.show()



#### Show more detailed data for the FMI Station "Kustavi Isokari"

# The times are as a list of datetime objects
times = obs.data['Kustavi Isokari']['times']
# Other data fields have another extra level, one for values and one for the unit
print(len(obs.data['Kustavi Isokari']['Air temperature']['values']))
# -> 71
print(obs.data['Kustavi Isokari']['Air temperature'])
# -> 'degC'
print(obs.data['Kustavi Isokari'])

