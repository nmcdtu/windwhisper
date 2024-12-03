"""
Fetch layer representing human settlements.
"""

import requests
from rasterio.io import MemoryFile
import xarray as xr
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

WMS_BASE_URL = os.getenv("API_WSF")

# Function to query WMS for a map preview
def get_wsf_map_preview(bbox, layers="WSF_2019", resolution=0.01):

    # Calculate width and height based on bbox dimensions and resolution
    lon_diff = bbox[2] - bbox[0]  # max_lon - min_lon
    lat_diff = bbox[3] - bbox[1]  # max_lat - min_lat

    width = int(lon_diff / resolution)
    height = int(lat_diff / resolution)

    params = {
        "service": "WMS",
        "request": "GetMap",
        "layers": layers,
        "bbox": ",".join(map(str, bbox)),
        "width": width,
        "height": height,
        "srs": "EPSG:4326",
        "crs": "EPSG:4326",
        "format": "image/png",
    }

    response = requests.get(WMS_BASE_URL, params=params)
    if response.status_code == 200:
        # Use a memory file to avoid saving to disk
        with MemoryFile(response.content) as memfile:
            with memfile.open() as dataset:
                # Read the first band of data
                data = dataset.read(1)
                data = (data > 0).astype("uint8")  # Convert to binary (1 = settlement, 0 = no settlement)

                # Generate latitude and longitude arrays based on bbox
                lat = np.linspace(bbox[1], bbox[3], height)  # Y-axis: min_lat to max_lat
                lon = np.linspace(bbox[0], bbox[2], width)  # X-axis: min_lon to max_lon

                return xr.DataArray(
                    data,
                    dims=["lat", "lon", ],
                    coords={
                        "lat": lat[::-1],
                        "lon": lon,
                    },
                    attrs={
                        "crs": "EPSG:4326",
                        "long_name": "World Settlement Footprint",
                        "units": "boolean",
                        "bbox": bbox,
                    }
                )
