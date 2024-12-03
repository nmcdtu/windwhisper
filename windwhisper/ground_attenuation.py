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

    # elevated distance is the distance between
    # the source and receiver, considering their
    # respective elevations.

    for longitude in longitudes:
        for latitude in latitudes:
            for turbine, specs in wind_turbines.items():
                source_position = (latitude, longitude)
                receiver_position = specs["position"]
                source_elevation = elevation_grid.interp(coords={"lat": latitude, "lon": longitude}).values.item(0)
                receiver_elevation = elevation_grid.interp(coords={"lat": receiver_position[0], "lon": receiver_position[1]}).values.item(0)
                elevated_distance = np.sqrt(haversine(source_position, receiver_position, unit=Unit.METERS) ** 2 + (source_elevation - receiver_elevation) ** 2)
                euclidian_distances.loc[dict(lat=latitude, lon=longitude, turbine=turbine)] = elevated_distance
                # path between source and receiver
                path =
                # calculate ground profile of the path between source and receiver
                ground_area_profile =


    # h_m in ISO 9613-2:2024 is the mean height of the ground profile
    #mean_height = ground_area_profile / elevated_distance
    #ground_attenuation = 4.8 - (2 * mean_height / distance) * (17 + (300 / distance))

    return elevation_grid, relative_elevations, euclidian_distances





def create_elevation_mask(source_coord, grid_coords, elevation_data):
    """
    Create a mask that disables coordinates blocked by elevation changes.

    Parameters:
        source_coord (tuple): Coordinates of the noise source (lon, lat).
        grid_coords (tuple): Tuple of 1D arrays representing the grid (lon_array, lat_array).
        elevation_data (2D array): Elevation data corresponding to the grid.

    Returns:
        mask (2D array): Mask with 1 where noise can propagate, 0 where it is blocked.
    """
    lon_array, lat_array = grid_coords
    source_lon, source_lat = source_coord

    # Initialize the mask with ones (everything unblocked)
    mask = np.ones_like(elevation_data, dtype=int)

    # Create an interpolator for the elevation data
    interpolator = RegularGridInterpolator((lat_array, lon_array), elevation_data)

    # Source elevation
    source_elevation = interpolator([source_lat, source_lon])

    # Iterate over each grid cell
    for i, lat in enumerate(lat_array):
        for j, lon in enumerate(lon_array):
            # Skip the source point itself
            if (lon, lat) == source_coord:
                continue

            # Receiver elevation
            receiver_elevation = elevation_data[i, j]

            # Interpolate along the path from source to receiver
            num_points = 100  # Number of points to sample along the path
            lats = np.linspace(source_lat, lat, num_points)
            lons = np.linspace(source_lon, lon, num_points)
            path_coords = np.column_stack((lats, lons))

            # Get elevations along the path
            path_elevations = interpolator(path_coords)

            # Check if any point along the path is higher than both the source and receiver
            if np.any(path_elevations > max(source_elevation, receiver_elevation)):
                mask[i, j] = 0  # Blocked

    return mask


# Example usage
lon_array = np.linspace(-10, 10, 100)  # Example longitude grid
lat_array = np.linspace(-10, 10, 100)  # Example latitude grid
elevation_data = np.random.rand(100, 100) * 100  # Example elevation data

source_coord = (0, 0)  # Source coordinates
mask = create_elevation_mask(source_coord, (lon_array, lat_array), elevation_data)
