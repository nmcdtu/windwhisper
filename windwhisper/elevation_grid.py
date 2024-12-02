"""
Fetch elevation grid for an area.
"""

import requests
import numpy as np
import xarray as xr


OPEN_ELEVATION_API = "https://api.open-elevation.com/api/v1/lookup?locations="

def get_elevation_grid(longitudes: list, latitudes: list) -> xr.DataArray:
    """
    Fetch elevation data for a given bounding box.

    :param: longitudes: A list of longitudes.
    :param: latitude: A list of latitudes.
    :return: A 2D numpy array containing the elevation data.
    """

    # determine the bounding box
    min_lon, max_lon = longitudes[0], longitudes[-1]
    min_lat, max_lat = latitudes[0], latitudes[-1]

    # sample 10 points along the bounding box
    lon_step = (max_lon - min_lon) / 10
    lat_step = (max_lat - min_lat) / 10

    longitudes_ = np.arange(min_lon, max_lon, lon_step)
    latitudes_ = np.arange(min_lat, max_lat, lat_step)

    # create a grid of latitudes and longitudes
    path = [(lat, lon) for lon in longitudes_ for lat in latitudes_]

    # get elevation data from Open Elevation API
    url = OPEN_ELEVATION_API
    for lat, lon in path:
        url += f"{lat},{lon}|"

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

        print(xr_array.isnull().sum())

        # fill missing values with the nearest neighbor
        xr_array = xr_array.interpolate_na(method="nearest", dim="lat")
        xr_array = xr_array.interpolate_na(method="nearest", dim="lon")

        print(xr_array.isnull().sum())


        return xr_array

    else:
        print(f"Failed to fetch elevation data: {response.status_code}")
        return None

