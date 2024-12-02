"""
Fetch layer representing human settlements.
"""

import requests
from rasterio.io import MemoryFile
import xarray as xr
import numpy as np
import rasterio
import os

WMS_BASE_URL = "https://geoservice.dlr.de/eoc/wms"


# Function to query WMS for a map preview
def get_wsf_map_preview(bbox, layers="WSF_2019"):

    width, height = 800, 600  # Define resolution
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
