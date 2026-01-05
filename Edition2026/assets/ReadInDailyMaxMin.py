#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:48:04 2025

@author: junga1
"""

import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, UTC
import json
import pandas as pd


def parse_xml(xml_string: str) -> dict:
    ns = {
        "wml2": "http://www.opengis.net/waterml/2.0",
        "gml": "http://www.opengis.net/gml/3.2",
        "target": "http://xml.fmi.fi/namespace/om/atmosphericfeatures/1.1",
        "om": "http://www.opengis.net/om/2.0",
    }

    root = ET.fromstring(xml_string)
    result = []

    for member in root.findall(
        ".//wfs:member", {"wfs": "http://www.opengis.net/wfs/2.0"}
    ):
        # Get location info
        location = member.find(".//target:Location", ns)
        station = location.find(
            './gml:name[@codeSpace="http://xml.fmi.fi/namespace/locationcode/name"]', ns
        ).text
        pos = member.find(".//gml:pos", ns).text.split()
        lat, lon = float(pos[0]), float(pos[1])

        param = (
            member.find(".//om:observedProperty", ns)
            .get("{http://www.w3.org/1999/xlink}href")
            .split("param=")[1]
            .split("&")[0]
        )

        for point in member.findall(".//wml2:point", ns):
            time = point.find(".//wml2:time", ns).text

            if datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ").hour != 0:
                # the wfs api does not allow setting of 24h timestep,
                # so we get multiple values per day (which are nan)
                continue

            value = point.find(".//wml2:value", ns).text

            result.append(
                {
                    "station": station,
                    "time": time,
                    "lat": lat,
                    "lon": lon,
                    "parameter": param,
                    "value": float(value) if value != "NaN" else None,
                }
            )

    return result


# Set query parameters
# Daily max and min temperature

parameters = ["tmax", "tmin"]

# Closest weather station is automatically selected
# Can also use station id (keyword fmisid: fmisid=101533&fmisid=101784) or bounding box (keyword bbox: bbox=21,61,23,63)
# https://en.ilmatieteenlaitos.fi/observation-stations

places = ["Helsinki", "Tampere"]

# Select past five days

end = (datetime.now(UTC) - timedelta(hours=24)).strftime("%Y-%m-%dT00:00:00Z")
start = (datetime.now(UTC) - timedelta(hours=5 * 24)).strftime("%Y-%m-%dT00:00:00Z")

# Get the data

places = "&".join(["place={}".format(place) for place in places])
parameters = ",".join(parameters)

stored_query = "fmi::observations::weather::daily::timevaluepair"
url = "http://opendata.fmi.fi/wfs?service=WFS&version=2.0.0&request=getFeature&storedquery_id={}&{}&starttime={}&endtime={}&parameters={}&timestep=720&".format(
    stored_query, places, start, end, parameters
)

########################

# Define the bounding box for Turku area
bbox = "bbox=22.2,60.3,23.0,60.7"  # Smaller bbox around Turku

#bbox = "bbox=25.6088,66.48832,25.709,66.49832"

bbox = "bbox=19.0,59.5,31.6,70.2" #entire finland

# Select past five days
end = (datetime.now(UTC) - timedelta(hours=24)).strftime("%Y-%m-%dT00:00:00Z")
start = (datetime.now(UTC) - timedelta(hours=5 * 24)).strftime("%Y-%m-%dT00:00:00Z")

# Specify the weather parameters
parameters = "tmin,tmax"  # Example parameters: minimum and maximum temperature

# Define the stored query
stored_query = "fmi::observations::weather::daily::timevaluepair"

# Construct the WFS query URL
url = (
    f"http://opendata.fmi.fi/wfs?service=WFS&version=2.0.0&request=getFeature&"
    f"storedquery_id={stored_query}&{bbox}&starttime={start}&endtime={end}&"
    f"parameters={parameters}&timestep=720"
)

print(url)


print(url)
response = requests.get(url)

assert response.status_code == 200, "Failed to fetch data: {}\nurl: {}".format(
    response.text, response.url
)

data = parse_xml(response.content)

json.dump(data, open("daily.json", "w"), indent=2)
print("Data saved to daily.json")

#data = json.loads(json_text)

# --- Long -> wide: one row per (station, day), columns per parameter (tmin/tmax/...) ---
df_long = pd.DataFrame(data)

required = {"station", "time", "lat", "lon", "parameter", "value"}
missing = required - set(df_long.columns)
if missing:
    raise KeyError(f"Missing keys in JSON records: {missing}")

# Normalize day column (keeps the original 'time' too if you want)
df_long["day"] = pd.to_datetime(df_long["time"], utc=True).dt.date.astype(str)

# Pivot
df_wide = (
    df_long.pivot_table(
        index=["station", "lat", "lon", "day"],
        columns="parameter",
        values="value",
        aggfunc="first",
    )
    .reset_index()
)

df_wide.columns.name = None  # drop pandas column index name

# Optional: stable column order (station meta + sorted parameters)
param_cols = sorted([c for c in df_wide.columns if c not in {"station", "lat", "lon", "day"}])
df_wide = df_wide[["station", "lat", "lon", "day"] + param_cols]

# --- Save ---
out_csv = "daily_records.csv"
df_wide.to_csv(out_csv, index=False)
print(f"Wrote {len(df_wide)} daily rows to {out_csv}")