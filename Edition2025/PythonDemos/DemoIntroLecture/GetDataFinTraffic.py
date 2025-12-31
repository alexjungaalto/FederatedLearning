import requests
import pandas as pd
import json
from PIL import Image
from io import BytesIO
import geopandas as gpd
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from shapely.geometry import Polygon, MultiPolygon

################
#### first we construct a polygon that resembles the shape of Finland 
####################
# Update the path to the location where you've saved the shapefile
shapefile_path = "ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
# Read the shapefile
world = gpd.read_file(shapefile_path)

# Grab Finland
finland_rows = world[(world["ADMIN"] == "Finland") | (world["SOVEREIGNT"] == "Finland")]
geometry = finland_rows["geometry"].iloc[0]

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
if coordinates and coordinates[0] != coordinates[-1]:
    coordinates.append(coordinates[0])

# Extract longitude and latitude
lons, lats = zip(*coordinates)

##########################
# Fetch all camera stations
##########################

camera_stations_url = "https://tie.digitraffic.fi/api/weathercam/v1/stations"
response = requests.get(camera_stations_url)

if response.status_code == 200:
    data = response.json()
    # Extract camera station information
    if data and "features" in data:
        camera_stations = data["features"]

        cameras = []
        for station in camera_stations:
            properties = station.get("properties", {})
            coords = station.get("geometry", {}).get("coordinates", [None, None])
            lon = coords[0] if len(coords) > 0 else None
            lat = coords[1] if len(coords) > 1 else None

            cameras.append(
                {
                    "Station ID": properties.get("id"),
                    "Name": properties.get("name", {}),
                    "Latitude": lat,
                    "Longitude": lon,
                    "Road Number": properties.get("roadNumber"),
                    "Municipality": properties.get("municipality"),
                    "Province": properties.get("province"),
                }
            )

        df_cameras = pd.DataFrame(cameras)
        df_cameras.to_csv("weather_camera_stations.csv", index=False)
        print("Weather camera stations data has been saved to 'weather_camera_stations.csv'.")
    else:
        print("No camera stations found in the data.")
else:
    print(f"Failed to fetch camera data: {response.status_code}")
    raise SystemExit()

#################################
# Example: fetch latest from one station (C01502) safely
#################################

test_station_id = "C01502"
url2 = f"https://tie.digitraffic.fi/api/weathercam/v1/stations/{test_station_id}/history"
response = requests.get(url2)

if response.status_code == 200:
    data = json.loads(response.content.decode("utf-8"))

    presets = data.get("presets") or []
    if not presets:
        print(f"No presets available for test station {test_station_id}.")
    else:
        history = presets[0].get("history") or []
        if not history:
            print(f"No history available for test station {test_station_id}.")
        else:
            latest_entry = max(history, key=lambda x: x["lastModified"])
            latest_image_url = latest_entry.get("imageUrl")

            if latest_image_url:
                try:
                    response = requests.get(latest_image_url)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                    # You can show or save here if you want
                    # plt.imshow(image); plt.axis("off"); plt.show()
                except requests.exceptions.RequestException as e:
                    print(f"Failed to fetch or display the image for {test_station_id}: {e}")
            else:
                print(f"No image URL in latest history entry for {test_station_id}.")
else:
    print(f"Failed to fetch history for test station {test_station_id}: {response.status_code}")

#################################
# Match each FMI station to nearest camera
#################################

weather_stations = pd.read_csv("fmi_stations_subset.csv")

nearest_cameras = []
for _, ws_row in weather_stations.iterrows():
    ws_coords = (ws_row["lat"], ws_row["lon"])
    df_cameras["Distance"] = df_cameras.apply(
        lambda row: geodesic(ws_coords, (row["Latitude"], row["Longitude"])).kilometers,
        axis=1,
    )
    nearest_camera = df_cameras.loc[df_cameras["Distance"].idxmin()].copy()
    nearest_camera["Weather Station"] = ws_row["Station"]
    nearest_cameras.append(nearest_camera)

df_nearest_cameras = pd.DataFrame(nearest_cameras)

#################################
# Prepare scatter plot with camera snapshots
#################################

fig, ax = plt.subplots(figsize=(12, 8))

for _, row in df_nearest_cameras.iterrows():
    station_id = row["Station ID"]
    latitude = row["Latitude"]
    longitude = row["Longitude"]
    print(f"Fetching latest snapshot for station: {station_id}")

    url = f"https://tie.digitraffic.fi/api/weathercam/v1/stations/{station_id}/history"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to fetch history for station {station_id}: {response.status_code}")
        continue

    try:
        data = json.loads(response.content.decode("utf-8"))
        presets = data.get("presets") or []
        if not presets:
            print(f"No presets available for station {station_id}.")
            continue

        history = presets[0].get("history") or []
        if not history:
            print(f"No history available for station {station_id}.")
            continue

        latest_entry = max(history, key=lambda x: x["lastModified"])
        latest_image_url = latest_entry.get("imageUrl")

        if not latest_image_url:
            print(f"No image URL available for station {station_id}.")
            continue

        image_response = requests.get(latest_image_url)
        image_response.raise_for_status()
        image = Image.open(BytesIO(image_response.content))

        # Shrink image for map
        img = OffsetImage(image, zoom=0.05)
        ab = AnnotationBbox(img, (longitude, latitude), frameon=False)
        ax.add_artist(ab)

        # Label with station name
        ax.text(
            longitude,
            latitude - 0.5,
            row["Name"],
            color="blue",
            fontsize=8,
            ha="center",
            va="top",
        )

    except Exception as e:
        print(f"Error fetching or plotting image for station {station_id}: {e}")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Weather Camera Snapshots")
ax.grid(True)

# Draw Finland polygon
ax.plot(lons, lats, marker="o", linestyle="-", markersize=3, label="Finland Border")

plt.tight_layout()
plt.show()
