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


##########################

# API endpoint for weather camera stations
camera_stations_url = "https://tie.digitraffic.fi/api/weathercam/v1/stations"

# Fetch the camera data
response = requests.get(camera_stations_url)

if response.status_code == 200:
    data = response.json()
    print(data)
    # Extract camera station information
    if data and "features" in data:
        camera_stations = data["features"]

        # Create a list of dictionaries for DataFrame
        cameras = []
        for station in camera_stations:
            properties = station.get("properties", {})
            cameras.append({
                "Station ID": properties.get("id"),
                "Name": properties.get("name", {}),
                "Latitude": station.get("geometry", {}).get("coordinates", [None, None])[1],
                "Longitude": station.get("geometry", {}).get("coordinates", [None, None])[0],
                "Road Number": properties.get("roadNumber"),
                "Municipality": properties.get("municipality"),
                "Province": properties.get("province"),
            })

        # Convert to DataFrame for easier analysis
        df_cameras = pd.DataFrame(cameras)

        # Save to a CSV file
        df_cameras.to_csv("weather_camera_stations.csv", index=False)
        print("Weather camera stations data has been saved to 'weather_camera_stations.csv'.")
    else:
        print("No camera stations found in the data.")
else:
    print(f"Failed to fetch camera data: {response.status_code}")



url2 = "https://tie.digitraffic.fi/api/weathercam/v1/stations/C01502/history"
response = requests.get(url2)

# Parse the response
data = json.loads(response.content.decode('utf-8'))

# Extract the latest image URL
history = data["presets"][0]["history"]
latest_entry = max(history, key=lambda x: x["lastModified"])
latest_image_url = latest_entry["imageUrl"]

# Download the image
try:
    # Fetch the image
    response = requests.get(latest_image_url)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)

    # Open the image
    image = Image.open(BytesIO(response.content))

    # Plot the image
#    plt.figure(figsize=(8, 6))
#    plt.imshow(image)
#    plt.axis('off')  # Hide axes for better visualization
#    plt.title("Latest Weather Camera Snapshot")
#    plt.show()

except requests.exceptions.RequestException as e:
    print(f"Failed to fetch or display the image: {e}")
    
weather_stations = pd.read_csv("fmi_stations_subset.csv")

nearest_cameras = []
for _, ws_row in weather_stations.iterrows():
   ws_coords = (ws_row["lat"], ws_row["lon"])
   df_cameras["Distance"] = df_cameras.apply(
                lambda row: geodesic(ws_coords, (row["Latitude"], row["Longitude"])).kilometers, axis=1
            )
   nearest_camera = df_cameras.loc[df_cameras["Distance"].idxmin()]
   nearest_camera["Weather Station"] = ws_row["Station"]
   nearest_cameras.append(nearest_camera)

df_nearest_cameras = pd.DataFrame(nearest_cameras)

# Prepare scatter plot
fig, ax = plt.subplots(figsize=(12, 8))

# Iterate over each camera station and retrieve the latest snapshot
for _, row in df_nearest_cameras.iterrows():
    station_id = row["Station ID"]
    latitude = row["Latitude"]
    longitude = row["Longitude"]
    print(f"Fetching latest snapshot for station: {station_id}")

# Fetch history for the current camera
    url = f"https://tie.digitraffic.fi/api/weathercam/v1/stations/{station_id}/history"
    response = requests.get(url)

    if response.status_code == 200:
        try:
# Parse the response
            data = json.loads(response.content.decode('utf-8'))

# Extract the latest image URL
            history = data.get("presets", [])[0].get("history", [])
            if history:
                latest_entry = max(history, key=lambda x: x["lastModified"])
                latest_image_url = latest_entry.get("imageUrl", None)

                if latest_image_url:
# Fetch and display the image
                    image_response = requests.get(latest_image_url)
                    image_response.raise_for_status()
                    image = Image.open(BytesIO(image_response.content))
# Resize the image
                    img = OffsetImage(image, zoom=0.05)  # Shrink image size for scatterplot
                    ab = AnnotationBbox(img, (longitude, latitude), frameon=False)
                    ax.add_artist(ab)

                            # Add a scatter point for the location
               #     ax.scatter(longitude, latitude, color="red", s=10, label=row["Name"])
                    # Add the station name below the snapshot
                    ax.text(longitude, latitude - 0.5, row["Name"], color="blue", 
                       fontsize=12, ha="center", va="top")  # Adjust the offset for positioning

                      
                else:
                    print(f"No image URL available for station {station_id}.")
            else:
                print(f"No history available for station {station_id}.")
        except Exception as e:
            print(f"Error fetching image for station {station_id}: {e}")
    else:
        print(f"Failed to fetch history for station {station_id}: {response.status_code}")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Weather Camera Snapshots")
ax.grid(True)
#plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Weather Stations")
plt.plot(lons, lats, marker='o', linestyle='-', markersize=5, label='Polygon Border')
plt.tight_layout()
plt.show()
