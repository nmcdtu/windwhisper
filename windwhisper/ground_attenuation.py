"""
Estimate noise attenuation due to ground type, distance from the source and elevation difference between the source
and receiver, according to ISO 9613-2:2024.
"""

import numpy as np
import requests
from haversine import haversine, Unit
import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .elevation_grid import get_elevation_grid, distances_with_elevation



OPEN_ELEVATION_API = "https://api.open-elevation.com/api/v1/lookup?locations="


def calculate_ground_attenuation(haversine_distances, longitudes, latitudes, wind_turbines):
    """
    Calculate the ground attenuation in dB, according to ISO 9613-2:2024.
    """

    elevation_grid = get_elevation_grid(
        longitudes=longitudes,
        latitudes=latitudes
    )

    if elevation_grid is None:
        return None, None, None

    relative_elevations = xr.DataArray(
        data=np.zeros((len(latitudes), len(longitudes), len(wind_turbines))),
        dims=("lat", "lon", "turbine"),
        coords={"lat": latitudes, "lon": longitudes, "turbine": list(wind_turbines.keys())}
    )

    for turbine, specs in wind_turbines.items():
        relative_elevations.loc[dict(turbine=turbine)] = elevation_grid.values - elevation_grid.interp(
            coords={"lat": specs["position"][0], "lon": specs["position"][1]}).values

    euclidian_distances = distances_with_elevation(haversine_distances, relative_elevations)
    euclidian_distances = xr.DataArray(
        data=euclidian_distances,
        dims=("lat", "lon", "turbine"),
        coords={"lat": latitudes, "lon": longitudes, "turbine": list(wind_turbines.keys())}
    )

    # Create an interpolator for the elevation data
    interpolator = RegularGridInterpolator(
        (latitudes, longitudes),
        elevation_grid.values
    )

    ground_attenuation = xr.DataArray(
        data=np.zeros((len(latitudes), len(longitudes), len(wind_turbines))),
        dims=("lat", "lon", "turbine"),
        coords={"lat": latitudes, "lon": longitudes, "turbine": list(wind_turbines.keys())}
    )

    obstacles_attenuation = xr.DataArray(
        data=np.ones((len(latitudes), len(longitudes), len(wind_turbines))),
        dims=("lat", "lon", "turbine"),
        coords={"lat": latitudes, "lon": longitudes, "turbine": list(wind_turbines.keys())}
    )

    for turbine, specs in wind_turbines.items():
        source_lat, source_lon = specs["position"]
        # Source elevation
        source_elevation = interpolator([source_lat, source_lon])

        # Iterate over each grid cell
        for i, lat in enumerate(latitudes):
            for j, lon in enumerate(longitudes):
                # Skip the source point itself
                if (lon, lat) == (source_lon, source_lat):
                    continue

                # also, skip if that coordinate has already been calculated
                if ground_attenuation.sel(lat=lat, lon=lon, turbine=turbine) != 0:
                    continue

                # Receiver elevation
                receiver_elevation = elevation_grid.sel(
                    lat=lat, lon=lon
                )
                path_latitudes = np.linspace(source_lat, lat, 100)
                path_longitudes = np.linspace(source_lon, lon, 100)
                path_coords = np.column_stack((path_latitudes, path_longitudes))

                # Get elevations along the path (ground profile)
                path_elevations = interpolator(path_coords)

                # calculate elevation along a straight line between source and receiver
                straight_elevation = np.squeeze(np.linspace(source_elevation, receiver_elevation, 100))

                # Check for obstacles
                obstacle_mask = path_elevations > (straight_elevation + 10)  # 10 m above the straight line
                obstacle_heights = path_elevations[obstacle_mask] - straight_elevation[obstacle_mask]

                if obstacle_heights.size > 0:
                    # Calculate obstacle attenuation
                    max_obstacle_height = obstacle_heights.max()
                    obstacle_distance = np.argmax(obstacle_mask) / len(path_coords) * euclidian_distance

                    # ISO 9613-2 obstacle attenuation formula
                    obstacle_attenuation = 10 + 20 * np.log10(max_obstacle_height / obstacle_distance)
                    obstacle_attenuation = np.clip(obstacle_attenuation, a_min=0, a_max=None)

                    obstacles_attenuation.loc[dict(lat=lat, lon=lon, turbine=turbine)] = obstacle_attenuation

                # calculate area between the path and the straight line
                # term F in ISO 9613-2:2024
                area = np.clip(
                    np.trapz(straight_elevation - path_elevations, dx=1),
                    a_min=0,
                    a_max=None
                )

                # term d_g in ISO 9613-2:2024
                euclidian_distance = euclidian_distances.sel(
                    lat=lat, lon=lon, turbine=turbine
                ).values.item()

                # term h_m in ISO 9613-2:2024
                mean_height = area / euclidian_distance

                # term A_gr in ISO 9613-2:2024
                attenuation = 4.8 - ((2 * mean_height)/ euclidian_distance) * (17 + (300 / euclidian_distance))
                attenuation = np.clip(attenuation, a_min=0, a_max=None)
                ground_attenuation.loc[dict(lat=lat, lon=lon, turbine=turbine)] = attenuation

    # sum the ground attenuations over turbines dimension
    # since those are dBs, we pick the maximum value
    ground_attenuation = ground_attenuation.max(dim="turbine")

    # sum the obstacles attenuations over turbines dimension
    # since those are booleans (0 = unreachable by soundwaves),
    # we pick the minimum value
    obstacles_attenuation = obstacles_attenuation.min(dim="turbine")

    return elevation_grid, ground_attenuation, obstacles_attenuation
