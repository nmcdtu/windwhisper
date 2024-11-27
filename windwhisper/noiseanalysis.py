import numpy as np
import xarray as xr
import folium
from pathlib import Path
from datetime import datetime
from typing import Dict

from . import DATA_DIR
from .utils import translate_4326_to_3035, create_bounding_box



class NoiseAnalysis:
    """
    This class handles the basic functionalities related to noise data analysis.

    :ivar wind_turbines: A list of dictionaries containing the wind turbine data.
    :ivar noise_map: A NoiseMap object containing the noise data.
    :ivar listeners: A list of dictionaries containing the observation points data.
    :ivar alpha: Air absorption coefficient.

    """

    def __init__(self, noise_map, wind_turbines, listeners):
        self.noise_map = noise_map
        self.wind_turbines = wind_turbines
        self.listeners = listeners
        self.lden_map = self.compute_lden()

    def compute_lden(self):
        """
        Compute Lden values from self.noise_level_at_mean_wind_speed and update the DataArray.

        Returns:
            xr.DataArray: Updated DataArray containing Lden values.
        """
        # Assuming the noise map has a 'time' coordinate for hourly noise levels
        noise = self.noise_map.noise_level_at_mean_wind_speed  # Noise levels as DataArray

        

        # Define time ranges for day, evening, and night
        day_mask = (noise["time"] >= 7) & (noise["time"] < 19)  # 07:00–19:00
        evening_mask = (noise["time"] >= 19) & (noise["time"] < 23)  # 19:00–23:00
        night_mask = (noise["time"] >= 23) | (noise["time"] < 7)  # 23:00–07:00

        # Convert noise levels to linear scale and apply weightings
        day_linear = 10 ** (noise.where(day_mask).mean(dim="time") / 10) * 12
        evening_linear = 10 ** ((noise.where(evening_mask).mean(dim="time") + 5) / 10) * 4
        night_linear = 10 ** ((noise.where(night_mask).mean(dim="time") + 10) / 10) * 8

        # Combine weighted intensities and compute Lden
        total_linear = (day_linear + evening_linear + night_linear) / 24
        lden = 10 * np.log10(total_linear)

        lden_map = noise.copy(data=lden)
        lden_map.attrs["long_name"] = "Lden Noise Levels"
        lden_map.attrs["units"] = "dB"

        return lden_map

