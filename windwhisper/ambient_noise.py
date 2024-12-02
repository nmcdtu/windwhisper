"""
This module fetches existing sources of noise from teh EU Noise maps, to figure out whether
implementing one or several wind turbines in a given area would be a net contribution to the
ambient noise level or not. Source: https://www.eea.europa.eu/en/datahub/datahubitem-view/c952f520-8d71-42c9-b74c-b7eb002f939b
"""
from typing import Tuple, Any

import requests
from rasterio.io import MemoryFile
import numpy as np
from pyproj import Transformer
import xarray as xr

from windwhisper.utils import create_bounding_box, translate_4326_to_3035

NOISE_MAPS_URLS = {
    "airports": "https://noise.discomap.eea.europa.eu/arcgis/rest/services/noiseStoryMap/NoiseContours_air_lden/ImageServer/exportImage",
    "industry": "https://noise.discomap.eea.europa.eu/arcgis/rest/services/noiseStoryMap/NoiseContours_ind_lden/ImageServer/exportImage",
    "highways": "https://noise.discomap.eea.europa.eu/arcgis/rest/services/noiseStoryMap/NoiseContours_road_lden/ImageServer/exportImage",
    "railtracks": "https://noise.discomap.eea.europa.eu/arcgis/rest/services/noiseStoryMap/NoiseContours_rail_lden/ImageServer/exportImage"
}

PIXEL_VALUE_TO_LDEN = {
    1: 55,
    2: 60,
    3: 65,
    4: 70,
    5: 75,
    15: 0,
    None: np.nan
}


def get_noise_values(url: str, x_min, x_max, y_min, y_max) -> xr.DataArray | None:
    params = {
        "bbox": f"{x_min},{y_min},{x_max},{y_max}",
        "bboxSR": "3035",
        "size": "800,600",  # Output resolution
        "format": "tiff",  # Request GeoTIFF
        "f": "image"  # Response type
    }

    # Fetch the GeoTIFF file
    response = requests.get(url, params=params)

    if response.status_code == 200:
        # Use a memory file to avoid saving to disk
        with MemoryFile(response.content) as memfile:
            with memfile.open() as dataset:
                # Read the first band of data
                data = dataset.read(1)
                data = np.vectorize(PIXEL_VALUE_TO_LDEN.get)(data)
                return data
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None


def combine_noise_levels(noise_layers: list) -> np.ndarray:
    """
    Combine noise levels from multiple sources in Lden.
    :param noise_layers: A list of numpy arrays with noise levels in Lden.
    :return: A numpy array with the combined noise levels in Lden.

    """
    # Convert Lden to linear scale
    linear_sum = np.sum([10 ** (layer / 10) for layer in noise_layers if layer is not None], axis=0)

    # Convert back to Lden
    combined = 10 * np.log10(linear_sum)

    return combined


def get_ambient_noise_levels(lon_min: float, lon_max: float, lat_min: float, lat_max: float) -> xr.DataArray | None:
    """
    Get the ambient noise levels for a given location.
    :param lon_min: Minimum longitude of the bounding box.
    :param lon_max: Maximum longitude of the bounding box.
    :param lat_min: Minimum latitude of the bounding box.
    :param lat_max: Maximum latitude of the bounding box.
    :return: A numpy array with the ambient noise levels in Lden.
    """

    noise_layers = []
    x_min, y_min = translate_4326_to_3035(lon_min, lat_min)
    x_max, y_max = translate_4326_to_3035(lon_max, lat_max)

    for t, url in NOISE_MAPS_URLS.items():
        layer = get_noise_values(url, x_min, x_max, y_min, y_max)

        layer = np.where(layer == None, 0, layer)  # Convert None to 0
        layer = layer.astype(float)

        if layer is not None:
            noise_layers.append(layer)

    if noise_layers:
        data = combine_noise_levels(noise_layers)

        # Example usage with combined_noise and bounding box
        resolution = (500 / 800)  # Calculate pixel size (bbox width divided by image width)
        return create_xarray_from_raster(
            data,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        )

    else:
        return None


def create_xarray_from_raster(data, x_min, y_min, x_max, y_max):
    """
    Create an xarray.DataArray from raster data and transform coordinates to EPSG:4326.

    Parameters:
        data (numpy.ndarray): The 2D array of raster data.
        x_min (float): Minimum X coordinate in EPSG:3035.
        y_min (float): Minimum Y coordinate in EPSG:3035.
        x_max (float): Maximum X coordinate in EPSG:3035.
        y_max (float): Maximum Y coordinate in EPSG:3035.
        resolution (float): Pixel resolution in the same units as x/y (e.g., meters).

    Returns:
        xarray.DataArray: Raster data as an xarray with longitude and latitude coordinates in EPSG:4326.
    """
    # Calculate the original x and y coordinates in EPSG:3035
    x_coords_3035 = np.linspace(x_min, x_max, data.shape[1])  # Columns
    y_coords_3035 = np.linspace(y_max, y_min, data.shape[0])  # Rows (reverse to match raster grid)

    # Transform coordinates to EPSG:4326
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon_coords, lat_coords = np.meshgrid(x_coords_3035, y_coords_3035)
    lon_coords, lat_coords = transformer.transform(lon_coords, lat_coords)

    # Create the DataArray
    raster_da = xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={
            "lat": lat_coords[:, 0],
            "lon": lon_coords[0, :],
        },
        attrs={
            "crs": "EPSG:4326",
            "long_name": "Combined Noise Levels",
            "units": "Lden (dB)"
        }
    )

    return raster_da
