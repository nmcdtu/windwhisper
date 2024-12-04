"""
Fetch elevation grid for an area.
"""

import requests
import numpy as np
import xarray as xr
import os
from dotenv import load_dotenv
from xarray import DataArray

load_dotenv()

OPEN_ELEVATION_API = os.getenv("API_OPEN_ELEVATION")
MAX_SAMPLING_POINTS = int(os.getenv("MAX_SAMPLING_POINTS"))

def get_elevation_grid(longitudes: np.array, latitudes: np.array) -> DataArray | None:
    """
    Fetch elevation data for a given bounding box.

    :param: longitudes: A list of longitudes.
    :param: latitude: A list of latitudes.
    :return: A 2D numpy array containing the elevation data.
    """

    # determine the bounding box
    min_lon, max_lon = min(longitudes), max(longitudes)
    min_lat, max_lat = min(latitudes), max(latitudes)

    # sample 10 points along the bounding box
    lon_step = (max_lon - min_lon) / MAX_SAMPLING_POINTS
    lat_step = (max_lat - min_lat) / MAX_SAMPLING_POINTS

    longitudes_ = np.arange(min_lon, max_lon + lon_step, lon_step)
    latitudes_ = np.arange(min_lat, max_lat + lat_step, lat_step)

    # create a grid of latitudes and longitudes
    path = [(lat, lon) for lon in longitudes_ for lat in latitudes_]

    # get elevation data from Open Elevation API
    url = OPEN_ELEVATION_API
    for lat, lon in path:
        url += f"{lat},{lon}|"  # append the coordinates to the URL
    url = url[:-1]

    response = requests.get(url)

    if response.status_code == 200:
        elevations = response.json()["results"]
        elevations = [elevation["elevation"] for elevation in elevations]

        # reshape the elevation data into a 2D array
        elevation_grid = np.array(elevations).reshape(len(latitudes_), len(longitudes_))

        xr_array = xr.DataArray(
            data=elevation_grid,
            dims=("lat", "lon"),
            coords={"lat": latitudes_, "lon": longitudes_}
        )

        xr_array.attrs["units"] = "meters"
        xr_array.attrs["description"] = "Elevation data for the given bounding box"
        xr_array.attrs["cs_code"] = "EPSG:4326"

        # increase resolution to match the original bounding box
        xr_array = xr_array.interp(lat=latitudes, lon=longitudes, method="linear")

        # fill missing values with the nearest neighbor
        xr_array = xr_array.interpolate_na(method="nearest", dim="lat")
        xr_array = xr_array.interpolate_na(method="nearest", dim="lon")

        # replace missing values with the mean of the surrounding values
        xr_array = xr_array.fillna(xr_array.mean())

        return xr_array

    else:
        print(f"Failed to fetch elevation data: {response.status_code}")
        return None

def distances_with_elevation(distances, relative_elevations):
    """
    Based on elevation and haversine distance, calculate the distance between two points
    considering the elevation difference.
    """
    return np.sqrt(distances ** 2 + relative_elevations ** 2)

