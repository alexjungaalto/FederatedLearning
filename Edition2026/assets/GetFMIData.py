#!/usr/bin/env python3
"""
Bulk-download FMI weather observations using fmiopendata's reference approach.

This script is designed for course projects where weather stations naturally
play the role of FL clients.

It uses the FMI WFS stored query:
    fmi::observations::weather::multipointcoverage

Two data layouts are supported:

(1) Snapshot layout (default: --timeseries not set)
    obs.data[time][station][parameter] = {'value': <float>, 'units': <str>}
    - Good when you want to treat each time step as a “round” or compute
      aggregates across all stations at a given time.

(2) Timeseries layout (--timeseries)
    obs.data[station]['times'] -> list[datetime]
    obs.data[station][parameter]['values'] -> list[float]
    obs.data[station][parameter]['unit'] -> str
    - Recommended for pandas/time-series analysis and for building per-station
      local datasets.

Output:
    A single CSV file in “long format” with columns:
        station, fmisid, lat, lon, time_utc, parameter, value, unit

Why “long format”?
    It is convenient for filtering, grouping, and pivoting in pandas.
    Example: df[df["parameter"]=="Air temperature"] or pivot tables.

Notes / limitations:
- multipointcoverage often returns *many* parameters; fine-grained parameter
  selection is more limited than in the timevaluepair query.
- For large time intervals, ALWAYS chunk the request (see --chunk-minutes),
  otherwise the response can become huge and/or fail.
"""

import argparse
import datetime as dt
import os
from typing import Dict, Any, Iterable, List

import pandas as pd
from fmiopendata.wfs import download_stored_query


# -----------------------------
# Small helper utilities
# -----------------------------

def iso_z(t: dt.datetime) -> str:
    """
    Convert a datetime to FMI-friendly ISO8601 string with a trailing 'Z'.

    FMI expects timestamps like: 2025-01-01T00:00:00Z

    If `t` is timezone-naive, we assume it is already UTC.
    """
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    t = t.astimezone(dt.timezone.utc)
    return t.isoformat(timespec="seconds").replace("+00:00", "Z")


def ensure_utc_naive_to_utc(dt_in: dt.datetime) -> dt.datetime:
    """
    Ensure datetime is in UTC timezone.

    - If dt_in is naive, assume UTC (common when users paste ISO times).
    - If dt_in has a timezone, convert to UTC.
    """
    if dt_in.tzinfo is None:
        return dt_in.replace(tzinfo=dt.timezone.utc)
    return dt_in.astimezone(dt.timezone.utc)


def iter_chunks(start: dt.datetime, end: dt.datetime, chunk_minutes: int) -> Iterable[tuple[dt.datetime, dt.datetime]]:
    """
    Yield consecutive half-open intervals [chunk_start, chunk_end)
    that cover [start, end).

    Example:
        start=00:00, end=12:00, chunk_minutes=360
        -> [00:00,06:00), [06:00,12:00)
    """
    cur = start
    delta = dt.timedelta(minutes=chunk_minutes)
    while cur < end:
        nxt = min(cur + delta, end)
        yield cur, nxt
        cur = nxt


# -----------------------------
# FMI query
# -----------------------------

def multipoint_query(bbox: str, start: str, end: str, timeseries: bool) -> Any:
    """
    Run FMI multipointcoverage query via fmiopendata.

    Parameters
    ----------
    bbox:
        String "minLon,minLat,maxLon,maxLat" (EPSG:4326 / WGS84).
        A Finland-ish bbox often used in exercises: "19,59,32,72"

    start, end:
        ISO8601 strings with trailing Z, e.g. "2025-01-01T00:00:00Z"

    timeseries:
        Whether to ask fmiopendata to restructure output by station.
    """
    args = [f"bbox={bbox}", f"starttime={start}", f"endtime={end}"]
    if timeseries:
        args.append("timeseries=True")

    return download_stored_query(
        "fmi::observations::weather::multipointcoverage",
        args=args
    )


# -----------------------------
# Converters: FMI object -> rows
# -----------------------------

def rows_from_timeseries(obs: Any) -> List[Dict[str, Any]]:
    """
    Convert timeseries=True structure into long rows.

    obs.data structure (timeseries=True):
        obs.data[station]['times'] -> list[datetime]
        obs.data[station][param]['values'] -> list
        obs.data[station][param]['unit'] -> str

    Returns
    -------
    rows:
        List of dictionaries matching CSV columns.
    """
    rows: List[Dict[str, Any]] = []
    meta = obs.location_metadata  # station_name -> {fmisid, latitude, longitude, ...}

    for station_name, station_data in obs.data.items():
        times: List[dt.datetime] = station_data.get("times", [])
        if not times:
            continue

        m = meta.get(station_name, {})
        fmisid = m.get("fmisid")
        lat = m.get("latitude")
        lon = m.get("longitude")

        # All keys except 'times' correspond to parameters.
        for param, payload in station_data.items():
            if param == "times":
                continue

            values = payload.get("values", [])
            unit = payload.get("unit")

            # Sometimes lengths mismatch; be safe.
            n = min(len(times), len(values))
            for i in range(n):
                t = times[i]
                v = values[i]
                rows.append({
                    "station": station_name,
                    "fmisid": fmisid,
                    "lat": lat,
                    "lon": lon,
                    "time_utc": iso_z(ensure_utc_naive_to_utc(t)),
                    "parameter": param,
                    "value": v,
                    "unit": unit,
                })

    return rows


def rows_from_snapshot(obs: Any) -> List[Dict[str, Any]]:
    """
    Convert default structure (time -> station -> parameter) into long rows.

    obs.data structure (timeseries=False):
        obs.data[time][station][param] = {'value': ..., 'units': ...}

    Returns
    -------
    rows:
        List of dictionaries matching CSV columns.
    """
    rows: List[Dict[str, Any]] = []
    meta = obs.location_metadata

    for t, by_station in obs.data.items():
        t_utc = iso_z(ensure_utc_naive_to_utc(t))

        for station_name, by_param in by_station.items():
            m = meta.get(station_name, {})
            fmisid = m.get("fmisid")
            lat = m.get("latitude")
            lon = m.get("longitude")

            for param, payload in by_param.items():
                rows.append({
                    "station": station_name,
                    "fmisid": fmisid,
                    "lat": lat,
                    "lon": lon,
                    "time_utc": t_utc,
                    "parameter": param,
                    "value": payload.get("value"),
                    "unit": payload.get("units"),
                })

    return rows


# -----------------------------
# CLI entry point
# -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Download FMI weather observations (multipointcoverage) and save as a long-format CSV."
    )

    ap.add_argument(
        "--bbox",
        default="19,59,32,72",
        help=("Bounding box 'minLon,minLat,maxLon,maxLat' (WGS84). "
              "Default is Finland-ish: 19,59,32,72")
    )
    ap.add_argument(
        "--start",
        required=True,
        help="Start time in ISO format, e.g. 2025-01-01T00:00:00Z"
    )
    ap.add_argument(
        "--end",
        required=True,
        help="End time in ISO format, e.g. 2025-01-02T00:00:00Z"
    )
    ap.add_argument(
        "--timeseries",
        action="store_true",
        help=("Organize response as station-centric time series "
              "(recommended for pandas and per-client datasets).")
    )
    ap.add_argument(
        "--out",
        default="fmi_obs.csv",
        help="Output CSV path (long format). Default: fmi_obs.csv"
    )
    ap.add_argument(
        "--chunk-minutes",
        type=int,
        default=0,
        help=("If >0, split [start,end] into chunks of this many minutes. "
              "Use this for long intervals to avoid huge responses, e.g. 360 (6h) or 1440 (1 day).")
    )
    ap.add_argument(
        "--print-stations",
        action="store_true",
        help="Print station names and metadata found in each response."
    )

    args = ap.parse_args()

    # Parse ISO timestamps. Support Z by converting to +00:00 for fromisoformat().
    start_dt = ensure_utc_naive_to_utc(dt.datetime.fromisoformat(args.start.replace("Z", "+00:00")))
    end_dt = ensure_utc_naive_to_utc(dt.datetime.fromisoformat(args.end.replace("Z", "+00:00")))

    if end_dt <= start_dt:
        raise ValueError("--end must be strictly after --start")

    all_rows: List[Dict[str, Any]] = []

    # Decide whether to chunk requests.
    if args.chunk_minutes and args.chunk_minutes > 0:
        chunks = list(iter_chunks(start_dt, end_dt, args.chunk_minutes))
    else:
        chunks = [(start_dt, end_dt)]

    # Process each chunk independently and append results.
    for (cs, ce) in chunks:
        cs_s = iso_z(cs)
        ce_s = iso_z(ce)

        print(f"Querying bbox={args.bbox} start={cs_s} end={ce_s} timeseries={args.timeseries}")

        obs = multipoint_query(args.bbox, cs_s, ce_s, args.timeseries)

        if args.print_stations:
            print("\nStations in response:")
            for name, m in sorted(obs.location_metadata.items(), key=lambda kv: kv[0]):
                print(f" - {name} | fmisid={m.get('fmisid')} | lat={m.get('latitude')} lon={m.get('longitude')}")
            print()

        # Convert to long-format rows.
        if args.timeseries:
            rows = rows_from_timeseries(obs)
        else:
            rows = rows_from_snapshot(obs)

        print(f"  extracted rows: {len(rows)}")
        all_rows.extend(rows)

    # Convert to a DataFrame for sorting and CSV export.
    df = pd.DataFrame(all_rows)

    # Deterministic ordering helps with debugging and reproducibility.
    if not df.empty:
        df = df.sort_values(["station", "time_utc", "parameter"], kind="stable")

    # Ensure output directory exists (if user specified a path like data/out.csv).
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows to {args.out}")

    # Optional: quick sanity check info.
    if not df.empty:
        print(f"Unique stations: {df['station'].nunique()}")
        print(f"Unique parameters: {df['parameter'].nunique()}")
        print("Example parameters:", list(df["parameter"].dropna().unique()[:10]))


if __name__ == "__main__":
    main()


"""
USAGE EXAMPLES
==============

1) Quick test: last hour for Finland bbox, station-centric timeseries output
---------------------------------------------------------------------------
python download_fmi_multipointcoverage.py \
  --start 2025-12-31T11:00:00Z \
  --end   2025-12-31T12:00:00Z \
  --timeseries \
  --print-stations \
  --out data/fmi_last_hour.csv

2) A full day, chunked into 6-hour queries (recommended for reliability)
-----------------------------------------------------------------------
python download_fmi_multipointcoverage.py \
  --start 2025-12-01T00:00:00Z \
  --end   2025-12-02T00:00:00Z \
  --timeseries \
  --chunk-minutes 360 \
  --out data/fmi_day.csv

3) Snapshot mode (time -> stations) for a short window
------------------------------------------------------
python download_fmi_multipointcoverage.py \
  --start 2025-12-31T00:00:00Z \
  --end   2025-12-31T03:00:00Z \
  --out data/fmi_snapshot.csv

TIP (pandas)
------------
import pandas as pd
df = pd.read_csv("data/fmi_day.csv")

# Filter one parameter, e.g. air temperature
temp = df[df["parameter"] == "Air temperature"]

# Convert time strings to pandas datetimes
temp["time_utc"] = pd.to_datetime(temp["time_utc"], utc=True)

# Pivot: rows=time, columns=station, values=temp
temp_pivot = temp.pivot_table(index="time_utc", columns="station", values="value")
"""
