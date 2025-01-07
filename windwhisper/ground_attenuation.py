"""
Estimate noise attenuation due to ground type, distance from the source and elevation difference between the source
and receiver, according to ISO 9613-2:2024.
"""


from multiprocessing import Pool
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from windwhisper.elevation_grid import get_elevation_grid, distances_with_elevation

# Move compute_turbine_attenuation outside
def compute_turbine_attenuation(args):
    turbine, specs, latitudes, longitudes, elevation_grid, interpolator, euclidian_distances = args
    source_lat, source_lon = specs["position"]
    source_elevation = interpolator([source_lat, source_lon])

    ground_attenuation = np.zeros((len(latitudes), len(longitudes)))
    obstacles_attenuation = np.zeros((len(latitudes), len(longitudes)))

    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            # Skip source point
            if (lon, lat) == (source_lon, source_lat):
                continue

            # Receiver elevation
            receiver_elevation = elevation_grid.values[i, j]
            path_latitudes = np.linspace(source_lat, lat, 100)
            path_longitudes = np.linspace(source_lon, lon, 100)
            path_coords = np.column_stack((path_latitudes, path_longitudes))

            # Elevation profile along the path
            path_elevations = interpolator(path_coords)
            straight_elevation = np.squeeze(np.linspace(source_elevation, receiver_elevation, path_elevations.size))

            # Obstacle detection
            obstacle_mask = path_elevations > (straight_elevation + 10)  # Boolean mask of obstacles

            if np.any(obstacle_mask):  # Check if there are any obstacles
                # Indexing is now safe because all arrays are 1D
                obstacle_heights = path_elevations[obstacle_mask] - straight_elevation[obstacle_mask]
                max_obstacle_height = obstacle_heights.max()

                # Calculate obstacle distance
                obstacle_distance = (
                        np.argmax(obstacle_mask) / len(path_elevations)
                        * euclidian_distances.values[i, j]
                )

                # ISO 9613-2 obstacle attenuation formula
                obstacle_attenuation = 10 + 20 * np.log10(max_obstacle_height / obstacle_distance)
                obstacles_attenuation[i, j] = max(0, obstacle_attenuation)

            # Area between path and straight line
            area = np.clip(np.trapz(straight_elevation - path_elevations, dx=1), 0, None)
            mean_height = area / euclidian_distances.values[i, j]
            attenuation = 4.8 - ((2 * mean_height) / euclidian_distances.values[i, j]) * (17 + (300 / euclidian_distances.values[i, j]))
            ground_attenuation[i, j] = max(0, attenuation)

    return ground_attenuation, obstacles_attenuation


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

    # Precompute interpolator
    interpolator = RegularGridInterpolator(
        (latitudes, longitudes),
        elevation_grid.values
    )

    # Precompute relative elevations
    relative_elevations = elevation_grid - xr.concat(
        [
            elevation_grid.interp(
                coords={"lat": specs["position"][0], "lon": specs["position"][1]}
            ).expand_dims(dim={"turbine": [turbine_name]})
            for turbine_name, specs in wind_turbines.items()
        ],
        dim="turbine",
    )

    euclidian_distances = distances_with_elevation(haversine_distances, relative_elevations)

    # Prepare arguments for parallel execution
    args = [
        (
            turbine,
            specs,
            latitudes,
            longitudes,
            elevation_grid,
            interpolator,
            euclidian_distances.sel(turbine=turbine)
        )
        for turbine, specs in wind_turbines.items()
    ]

    # Parallelize turbine computation
    with Pool() as pool:
        results = pool.map(compute_turbine_attenuation, args)

    # Aggregate results
    ground_attenuation = np.min([res[0] for res in results], axis=0)
    obstacles_attenuation = np.min([res[1] for res in results], axis=0)

    return elevation_grid, xr.DataArray(ground_attenuation, dims=("lat", "lon"), coords={"lat": latitudes, "lon": longitudes}), xr.DataArray(obstacles_attenuation, dims=("lat", "lon"), coords={"lat": latitudes, "lon": longitudes})
