import numpy as np
import xarray as xr
from .ambient_noise import get_ambient_noise_levels
from .settlement import get_wsf_map_preview

def create_bounding_box(noise_map):
    """
    Create a bounding box for the given noise map.

    :param noise_map: A DataArray containing the noise levels.
    :return: A tuple containing the bounding box coordinates.
    """
    lat_min = noise_map.lat.min().values.item(0)
    lat_max = noise_map.lat.max().values.item(0)
    lon_min = noise_map.lon.min().values.item(0)
    lon_max = noise_map.lon.max().values.item(0)

    return {
        "lat_min": lat_min,
        "lat_max": lat_max,
        "lon_min": lon_min,
        "lon_max": lon_max
    }

class NoiseAnalysis:
    """
    This class handles the basic functionalities related to noise data analysis.

    :ivar wind_turbines: A list of dictionaries containing the wind turbine data.
    :ivar noise_map: A NoiseMap object containing the noise data.
    :ivar alpha: Air absorption coefficient.

    """

    def __init__(self, noise_map, wind_turbines):
        self.noise_map = noise_map
        self.wind_turbines = wind_turbines
        self.lden_map = self.compute_lden()
        self.bbox = create_bounding_box(self.lden_map)
        self.ambient_noise_map = get_ambient_noise_levels(
            lon_min=self.bbox["lon_min"],
            lon_max=self.bbox["lon_max"],
            lat_min=self.bbox["lat_min"],
            lat_max=self.bbox["lat_max"]
        )

        self.settlement_map = get_wsf_map_preview(
            bbox=(
                self.bbox["lon_min"],
                self.bbox["lat_min"],
                self.bbox["lon_max"],
                self.bbox["lat_max"]
            )
        )

        self.merged_map = self.merge_maps()
        self.create_countours()

    def compute_lden(self):
        """
        Compute Lden values from self.noise_level_at_mean_wind_speed and update the DataArray.

        Returns:
            xr.DataArray: Updated DataArray containing Lden values.
        """
        # Assuming the noise map has a 'time' coordinate for hourly noise levels
        noise = self.noise_map.hourly_noise_levels  # Noise levels as DataArray

        # Define time ranges for day, evening, and night
        day_mask = (noise["hour"] >= 7) & (noise["hour"] < 19)  # 07:00–19:00
        evening_mask = (noise["hour"] >= 19) & (noise["hour"] < 23)  # 19:00–23:00
        night_mask = (noise["hour"] >= 23) | (noise["hour"] < 7)  # 23:00–07:00

        # Convert noise levels to linear scale and apply weightings
        day_linear = 10 ** (noise.where(day_mask).mean(dim="hour") / 10) * 12
        evening_linear = 10 ** ((noise.where(evening_mask).mean(dim="hour") + 5) / 10) * 4
        night_linear = 10 ** ((noise.where(night_mask).mean(dim="hour") + 10) / 10) * 8

        # Combine weighted intensities and compute Lden
        total_linear = (day_linear + evening_linear + night_linear) / 24
        lden = 10 * np.log10(total_linear)

        lden_map = noise.sel(hour=0).copy(data=lden)
        lden_map.attrs["long_name"] = "Lden Noise Levels"
        lden_map.attrs["units"] = "dB"

        return lden_map


    def merge_maps(self):

        # Define the target grid based on the higher resolution grid (ambient_noise_map)
        target_lat = self.ambient_noise_map.coords["lat"]
        target_lon = self.ambient_noise_map.coords["lon"]

        # Interpolate lden_map to the resolution of ambient_noise_map
        lden_interpolated = self.lden_map.interp(
            lat=target_lat,
            lon=target_lon,
            method="linear"
        )

        settlement_interpolated = self.settlement_map.interp(
            lat=target_lat,
            lon=target_lon,
            method="linear"
        )

        # Combine the two datasets into a single xarray
        merged_dataset = xr.Dataset({
            "ambient": self.ambient_noise_map,
            "wind": lden_interpolated,
            "settlement": settlement_interpolated
        })

        # Calculate the combined noise level (in dB) using the logarithmic formula
        noise_combined = 10 * np.log10(
            10 ** (merged_dataset["ambient"] / 10) + 10 ** (merged_dataset["wind"] / 10)
        )

        # Add the new layer to the dataset
        merged_dataset["combined"] = noise_combined

        # Add metadata for clarity
        merged_dataset["combined"].attrs["description"] = "Combined noise levels (ambient + LDEN) in dB"
        merged_dataset["combined"].attrs["units"] = "dB"

        # Calculate the net contribution of lden_noise to the combined noise level
        net_contribution = 10 * np.log10(
            10 ** (merged_dataset["combined"] / 10) / 10 ** (merged_dataset["ambient"] / 10)
        )

        # Add the net contribution layer to the dataset
        merged_dataset["net"] = net_contribution
        merged_dataset["net"].attrs["description"] = "Net contribution of LDEN noise levels in dB"

        return merged_dataset

    def create_countours(self):

        contour_30 = self.merged_map["combined"].where(self.merged_map["combined"] < 30, drop=True)
        contour_40 = (
            self.merged_map["combined"].where((self.merged_map["combined"] >= 30) & (self.merged_map["combined"] < 40), drop=True)
        )
        contour_50 = (
            self.merged_map["combined"].where((self.merged_map["combined"] >= 40) & (self.merged_map["combined"] < 50), drop=True)
        )
        contour_60 = (
            self.merged_map["combined"].where((self.merged_map["combined"] >= 50) & (self.merged_map["combined"] < 60), drop=True)
        )
        contour_70 = (
            self.merged_map["combined"].where((self.merged_map["combined"] >= 60) & (self.merged_map["combined"] < 70), drop=True)
        )
        contour_80 = self.merged_map["combined"].where(self.merged_map["combined"] >= 70, drop=True)


        # add it to the dataset
        self.merged_map["contour_30"] = contour_30 > 0
        self.merged_map["contour_40"] = contour_40 > 0
        self.merged_map["contour_50"] = contour_50 > 0
        self.merged_map["contour_60"] = contour_60 > 0
        self.merged_map["contour_70"] = contour_70 > 0
        self.merged_map["contour_80"] = contour_80 > 0


        # Condition 1: Ambient noise is below 50 dB
        ambient_below_50 = self.merged_map["ambient"] < 50

        # Condition 2: Combined noise exceeds 50 dB
        combined_above_50 = self.merged_map["combined"] > 50

        # Highlight coordinates where both conditions are true
        highlighted_coords = ambient_below_50 & combined_above_50

        # Add the highlighted coordinates as a layer in the dataset
        self.merged_map["flip"] = highlighted_coords

        # Add metadata for clarity
        self.merged_map["flip"].attrs["description"] = (
            "Coordinates where ambient noise < 50 dB but combined noise > 50 dB"
        )
        self.merged_map["flip"].attrs["datatype"] = "boolean"



