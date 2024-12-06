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
def get_wsf_map_preview(lon_min, lon_max, lat_min, lat_max, resolution, layers="WSF_2019"):
    """
    Fetch a map preview of the World Settlement Footprint (WSF) dataset.
    """

    width, height = resolution[1], resolution[0]

    bbox=f"{lon_min},{lat_min},{lon_max},{lat_max}"

    params = {
        "service": "WMS",
        "request": "GetMap",
        "layers": layers,
        "bbox": bbox,
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
                lat = np.linspace(lat_min, lat_max, height)
                lon = np.linspace(lon_min, lon_max, width)

                return xr.DataArray(
                    data,
                    dims=["lat", "lon", ],
                    coords={
                        "lat": lat,
                        "lon": lon,
                    },
                    attrs={
                        "crs": "EPSG:4326",
                        "long_name": "World Settlement Footprint",
                        "units": "boolean",
                        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
                    }
                )
