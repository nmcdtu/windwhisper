import numpy as np
from haversine import haversine, Unit
import ipywidgets as widgets
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import interp1d
from dotenv import load_dotenv
import os

from .atmospheric_absorption import get_absorption_coefficient
from .geometric_divergence import get_geometric_spread_loss
from .ground_attenuation import calculate_ground_attenuation

load_dotenv()

NOISE_MAP_RESOLUTION = int(os.getenv("NOISE_MAP_RESOLUTION", 100))
NOISE_MAP_MARGIN = float(os.getenv("NOISE_MAP_MARGIN", 0.02))

def define_bounding_box(wind_turbines: dict) -> tuple:
    """
    Define the bounding box for the noise map based on the wind turbines' positions.

    :param wind_turbines: A dictionary containing the wind turbine data.

    Returns:
        tuple: A tuple containing the bounding box coordinates.
    """

    # Determine the bounding box for the map

    lat_min = min(
        turbine["position"][0] for turbine in wind_turbines.values()
    )
    lat_max = max(
        turbine["position"][0] for turbine in wind_turbines.values()
    )
    lon_min = min(
        turbine["position"][1] for turbine in wind_turbines.values()
    )
    lon_max = max(
        turbine["position"][1] for turbine in wind_turbines.values()
    )

    # Adjust the map size to include observation points

    lat_min -= NOISE_MAP_MARGIN
    lat_max += NOISE_MAP_MARGIN
    lon_min -= NOISE_MAP_MARGIN
    lon_max += NOISE_MAP_MARGIN

    lon_array = np.linspace(lon_min, lon_max, NOISE_MAP_RESOLUTION)
    lat_array = np.linspace(lat_min, lat_max, NOISE_MAP_RESOLUTION)
    LON, LAT = np.meshgrid(lon_array, lat_array)

    return LAT[:, 0], LON[0, :]

class NoisePropagation:
    """
    The NoiseMap class is responsible for generating and displaying noise maps based on sound intensity levels.

    :ivar wind_turbines: A list of dictionaries containing the wind turbine data.
    :ivar noise: A xarray.DataArray containing the noise data vs wind speed.
    :ivar listeners: A list of dictionaries containing the observation points data.
    :ivar alpha: Air absorption coefficient.
    """

    def __init__(
        self,
        wind_turbines: dict,
        humidity: float = 70,
        temperature: float = 20,
    ):
        """
        Initialize the NoiseMap class.

        """
        self.temperature = temperature
        self.humidity = humidity
        self.wind_turbines = wind_turbines
        self.LAT, self.LON = define_bounding_box(wind_turbines)


        self.calculate_noise_attenuation_terms()

        self.noise_level_at_wind_speeds = self.noise_map_at_wind_speeds(
            np.vstack(
                [
                    specs["noise_vs_wind_speed"].values
                    for specs in self.wind_turbines.values()
                ]
            ),
            coord_name="wind_speed",
            coord_value=[specs["noise_vs_wind_speed"].coords["wind_speed"].values for specs in self.wind_turbines.values()][0],
        )

        self.calculate_hourly_noise_levels()

        self.hourly_noise_levels = self.noise_map_at_wind_speeds(
            np.vstack(
                [
                    specs["noise_per_hour"].values
                    for specs in self.wind_turbines.values()
                ]
            ),
            coord_name="hour",
            coord_value=[specs["noise_per_hour"].coords["hour"].values for specs in self.wind_turbines.values()][0],
        )


    def calculate_hourly_noise_levels(self):

        for turbine, turbine_specs in self.wind_turbines.items():
            wind_speeds = turbine_specs["mean_wind_speed"].values.flatten()
            noise_levels = turbine_specs["noise_vs_wind_speed"].values
            noise_level_wind_speeds = turbine_specs["noise_vs_wind_speed"].coords["wind_speed"].values

            # Create interpolation function
            interpolate_noise = interp1d(
                noise_level_wind_speeds,
                noise_levels,
                bounds_error=False,
                fill_value="extrapolate"
            )

            # Interpolate to find noise levels for the average wind speeds
            calculated_noise_levels = interpolate_noise(wind_speeds)

            # Create an xarray DataArray for the results
            noise_per_hour = xr.DataArray(
                calculated_noise_levels,
                dims=["hour"],
                coords={"hour": np.arange(len(wind_speeds))},
                name="noise_level"
            )

            # Add metadata
            noise_per_hour.attrs["units"] = "dB"
            noise_per_hour.attrs["description"] = "Predicted noise levels for hourly average wind speeds"

            # Add the noise levels to the wind turbine specs
            turbine_specs["noise_per_hour"] = noise_per_hour


    def noise_map_at_wind_speeds(self, noise, coord_name, coord_value) -> xr.DataArray:
        """
        Generates a noise map for the wind turbines
        and observation points for each wind speed level.

        :param noise: A 2D array representing the noise level vs wind speed.

        Returns:
            np.ndarray: A 2D array representing the noise map.
        """

        total_attenuation = (
                self.noise_attenuation.distance_attenuation +
                self.noise_attenuation.atmospheric_absorption
        )

        if "ground_attenuation" in self.noise_attenuation.data_vars:
            total_attenuation += self.noise_attenuation.ground_attenuation

        # replace NaN values with 0
        total_attenuation = total_attenuation.fillna(0)

        attenuated_noise = noise - total_attenuation.values[..., None, None]
        attenuated_noise = np.clip(attenuated_noise, a_min=0, a_max=None)

        # apply obstacle attenuation
        if "obstacle_attenuation" in self.noise_attenuation.data_vars:
            attenuated_noise *= self.noise_attenuation.obstacle_attenuation.values[..., None, None]

        # we need to sum the noise levels along the turbine axis
        Z = 10 * np.log10((10 ** (attenuated_noise / 10)).sum(axis=2))

        # create xarray to store Z
        Z = xr.DataArray(
            data=Z,
            dims=("lat", "lon", coord_name),
            coords={"lat": self.LAT, "lon": self.LON, coord_name: coord_value},
        )

        Z.values = np.clip(Z.values, a_min=0, a_max=None)

        return Z

    def calculate_noise_attenuation_terms(self):
        """
        Calculate the noise attenuation due to:
        * distance
        * atmospheric absorption
        * ground type
        * elevation difference between the source and receiver.

        """

        # Calculate the noise level at each point
        positions = [point["position"] for point in self.wind_turbines.values()]

        self.haversine_distances = np.array(
            [
                haversine(point1=(lat, lon), point2=position, unit=Unit.METERS)
                for lat in self.LAT
                for lon in self.LON
                for position in positions
            ]
        ).reshape(self.LAT.shape[0], self.LAT.shape[0], len(positions))

        # Calculate the geometric spreading loss, according to ISO 9613-2
        distance_attenuation = get_geometric_spread_loss(self.haversine_distances)
        # pick the minimum value along the turbine axis
        distance_attenuation = np.min(distance_attenuation, axis=-1)

        # Calculate the atmospheric absorption loss, according to ISO 9613-2
        atmospheric_absorption = (get_absorption_coefficient(
            self.temperature,
            self.humidity
        ) * self.haversine_distances) / 1000

        # pick the minimum value along the turbine axis
        atmospheric_absorption = np.min(atmospheric_absorption, axis=-1)

        # calculation elevation of grid cells compared to turbines' positions
        # we do this by subtracting the elevation of each grid cell from the elevation of the turbines
        self.elevation_grid, ground_attenuation, obstacle_attenuation = calculate_ground_attenuation(
            self.haversine_distances,
            self.LON,
            self.LAT,
            self.wind_turbines
        )

        data_vars = {}
        if ground_attenuation is not None:
            data_vars["ground_attenuation"] = (["lat", "lon"], ground_attenuation.data)
        if obstacle_attenuation is not None:
            data_vars["obstacle_attenuation"] = (["lat", "lon"], obstacle_attenuation.data)
        if distance_attenuation is not None:
            data_vars["distance_attenuation"] = (["lat", "lon"], distance_attenuation)
        if atmospheric_absorption is not None:
            data_vars["atmospheric_absorption"] = (["lat", "lon"], atmospheric_absorption)

        # Create an xarray Dataset
        self.noise_attenuation = xr.Dataset(
            data_vars,
            coords={
                "lat": self.LAT,
                "lon": self.LON,
            },
        )

    def plot_noise_map(self, dimension: str = "wind_speed"):
        """
        Plots the noise map with wind turbines and observation points.
        """

        # Create a wind speed slider for user interaction
        if dimension == "wind_speed":
            slider = widgets.FloatSlider(
                value=7.0,
                min=3.0,
                max=12.0,
                step=1.0,
                description="Wind Speed (m/s):",
                continuous_update=True,
            )
        else:
            # against hours of the day
            slider = widgets.IntSlider(
                value=12,
                min=0,
                max=23,
                step=1,
                description="Hour of the day:",
                continuous_update=True,
            )

        @widgets.interact(wind_speed=slider)
        def interactive_plot(wind_speed):
            plt.figure(figsize=(10, 6))

            if dimension == "wind_speed":
                data = self.noise_level_at_wind_speeds.interp(
                    wind_speed=wind_speed, kwargs={"fill_value": "extrapolate"}
                )
            else:
                data = self.hourly_noise_levels.interp(
                    hour=wind_speed, kwargs={"fill_value": "extrapolate"}
                )

            # Define contour levels starting from 35 dB
            contour_levels = [35, 40, 45, 50, 55, 60]

            # add bounding box
            plt.xlim(self.LON.min(), self.LON.max())
            plt.ylim(self.LAT.min(), self.LAT.max())

            plt.contourf(
                self.LON,  # x-axis, longitude
                self.LAT,  # y-axis, latitude
                data,
                levels=contour_levels,
                cmap="RdYlBu_r",
            )
            plt.colorbar(label="Noise Level (dB)")
            plt.title("Wind Turbine Noise Contours")
            plt.xlabel("Longitude")  # Correct label for x-axis
            plt.ylabel("Latitude")  # Correct label for y-axis

            # Plot wind turbines
            for turbine, specs in self.wind_turbines.items():
                plt.plot(
                    *specs["position"][::-1], "ko"
                )  # Make sure the position is in (Longitude, Latitude) order
                # add label next to it, add a small offset to avoid overlapping
                plt.text(
                    specs["position"][1] + 0.003,
                    specs["position"][0] + 0.002,
                    turbine,
                )

            plt.grid(True)
            plt.show()

