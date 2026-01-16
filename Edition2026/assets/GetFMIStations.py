

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch station metadata from FMI WFS and write stations.csv.

Outputs:
  stations.csv with columns: station_id, station, lat, lon

Notes:
- Uses storedquery_id = fmi::ef::stations (FMI "stations" query).
- Optionally you can set NETWORK_ID to limit the station network.
"""

from __future__ import annotations

import csv
import re
import sys
from typing import Optional, Iterable, Tuple, Dict, Any

import requests
import xml.etree.ElementTree as ET


WFS_URL = "https://opendata.fmi.fi/wfs"
STOREDQUERY_ID = "fmi::ef::stations"

# Optional: set to an integer (as string) to restrict network, else None
NETWORK_ID: Optional[str] = None  # e.g. "121" or "135"; leave None to fetch all

OUT_CSV = "stations.csv"
TIMEOUT_SEC = 60


NS = {
    "wfs": "http://www.opengis.net/wfs/2.0",
    "gml": "http://www.opengis.net/gml/3.2",
    "ef": "http://inspire.ec.europa.eu/schemas/ef/4.0",
    "om": "http://www.opengis.net/om/2.0",
    "xlink": "http://www.w3.org/1999/xlink",
}


def _text(node: Optional[ET.Element]) -> Optional[str]:
    if node is None:
        return None
    t = node.text
    return t.strip() if t else None


def _first_text(member: ET.Element, xpaths: Iterable[str]) -> Optional[str]:
    for xp in xpaths:
        n = member.find(xp, NS)
        t = _text(n)
        if t:
            return t
    return None


def _extract_fmisid(member: ET.Element) -> Optional[int]:
    # Preferred: gml:identifier with stationcode/fmisid
    t = _first_text(
        member,
        [
            './/gml:identifier[@codeSpace="http://xml.fmi.fi/namespace/stationcode/fmisid"]',
            './/gml:identifier[contains(@codeSpace,"stationcode") and contains(@codeSpace,"fmisid")]',
        ],
    )
    if t and t.isdigit():
        return int(t)

    # Fallback: parse from gml:id patterns like "...fmisid-101007..."
    # Search common elements that might carry gml:id
    for el in member.iter():
        gml_id = el.get("{http://www.opengis.net/gml/3.2}id")
        if gml_id:
            m = re.search(r"fmisid-(\d+)", gml_id)
            if m:
                return int(m.group(1))
    return None


def _extract_name(member: ET.Element) -> Optional[str]:
    # FMI sometimes uses ef:name, sometimes gml:name; keep it flexible.
    return _first_text(
        member,
        [
            ".//ef:name",
            ".//gml:name",
            './/gml:name[@codeSpace="http://xml.fmi.fi/namespace/locationcode/name"]',
        ],
    )


def _extract_lat_lon(member: ET.Element) -> Optional[Tuple[float, float]]:
    # Most common: a gml:pos somewhere in the feature
    pos_text = _first_text(member, [".//gml:pos"])
    if not pos_text:
        return None

    parts = pos_text.split()
    if len(parts) != 2:
        return None

    # FMI typically uses "lat lon" order in many outputs; we follow that convention.
    lat = float(parts[0])
    lon = float(parts[1])
    return lat, lon


def fetch_stations_xml() -> bytes:
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "storedquery_id": STOREDQUERY_ID,
    }
    if NETWORK_ID is not None:
        params["networkid"] = NETWORK_ID

    r = requests.get(WFS_URL, params=params, timeout=TIMEOUT_SEC)
    if r.status_code != 200:
        raise RuntimeError(f"WFS request failed ({r.status_code}): {r.text[:500]}")
    return r.content


def parse_stations(xml_bytes: bytes) -> list[Dict[str, Any]]:
    root = ET.fromstring(xml_bytes)
    rows: list[Dict[str, Any]] = []

    for member in root.findall(".//wfs:member", NS):
        station_id = _extract_fmisid(member)
        name = _extract_name(member)
        ll = _extract_lat_lon(member)

        # Keep only stations where we have all required fields
        if station_id is None or name is None or ll is None:
            continue

        lat, lon = ll
        rows.append(
            {
                "station_id": int(station_id),
                "station": name,
                "lat": float(lat),
                "lon": float(lon),
            }
        )

    # Deduplicate by station_id (keep first occurrence, but sanity-check consistency)
    rows_sorted = sorted(rows, key=lambda d: d["station_id"])
    out: dict[int, Dict[str, Any]] = {}
    for r in rows_sorted:
        sid = r["station_id"]
        if sid not in out:
            out[sid] = r
        else:
            # Consistency check: same id must map to same metadata (within rounding)
            prev = out[sid]
            if prev["station"] != r["station"]:
                raise ValueError(f"station_id {sid} has multiple names: '{prev['station']}' vs '{r['station']}'")
            if round(prev["lat"], 6) != round(r["lat"], 6) or round(prev["lon"], 6) != round(r["lon"], 6):
                raise ValueError(f"station_id {sid} has multiple coordinates: "
                                 f"({prev['lat']},{prev['lon']}) vs ({r['lat']},{r['lon']})")

    return list(out.values())


def write_csv(rows: list[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["station_id", "station", "lat", "lon"])
        w.writeheader()
        for r in sorted(rows, key=lambda d: d["station_id"]):
            w.writerow(r)


def main() -> None:
    xml_bytes = fetch_stations_xml()
    stations = parse_stations(xml_bytes)

    if not stations:
        msg = (
            "No stations were parsed.\n"
            "Possible causes:\n"
            " - FMI changed the XML structure.\n"
            " - networkid filter is too restrictive (try NETWORK_ID=None).\n"
        )
        sys.exit(msg)

    write_csv(stations, OUT_CSV)
    print(f"Wrote {len(stations)} stations to {OUT_CSV}")
    if NETWORK_ID is not None:
        print(f"(networkid={NETWORK_ID})")


if __name__ == "__main__":
    main()
