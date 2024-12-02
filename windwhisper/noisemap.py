import numpy as np
from haversine import haversine, Unit
import folium
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import interp1d

NOISE_MAP_RESOLUTION = 100

class NoiseMap:
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
        listeners: dict,
        alpha: float = 2.0,
    ):
        """
        Initialize the NoiseMap class.

        """
        self.alpha = alpha / 1000  # Convert alpha from dB/km to dB/m
        self.wind_turbines = wind_turbines
        self.listeners = listeners
        self.individual_noise = (
            self.superimpose_wind_turbines_noise()
        )  # pairs table with for each turbine, each listener

        self.LAT, self.LON, self.noise_attenuation = self.generate_noise_map()
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

    def calculate_sound_level_at_distance(
        self, dBsource: float, distance: float
    ) -> float:
        """
        Calculate the sound level at a given distance from the source considering
        attenuation due to distance and atmospheric absorption.

        Parameters:
            dBsource (float): Sound level in decibels at the source.
            distance (float): Distance from the source in meters.

        Returns:
            float: Sound level at the given distance in decibels.
        """
        if distance == 0:
            return dBsource

        geometric_spreading_loss = 10 * np.log10(4 * np.pi * distance**2) + 11
        atmospheric_absorption_loss = self.alpha * distance

        total_attenuation = geometric_spreading_loss + atmospheric_absorption_loss
        resulting_sound_level = dBsource - total_attenuation

        return resulting_sound_level

    def superimpose_wind_turbines_noise(self):
        """
        Superimposes the sound levels of several wind turbines
        :return: a list of dictionaries, with each dictionary representing a pair of turbine and listener
        and the distance between them and the sound level at that distance for
        each wind speed level
        """
        pairs = [
            {
                "turbine_name": turbine,
                "turbine_position": turbine_specs["position"],
                "listener_name": listener,
                "listener_position": listener_specs["position"],
                "distance": round(
                    haversine(
                        turbine_specs["position"], listener_specs["position"], unit=Unit.METERS
                    )
                ),
            }
            for turbine, turbine_specs in self.wind_turbines.items()
            for listener, listener_specs in self.listeners.items()
        ]

        # add dB level for each turbine
        for p, pair in enumerate(pairs):
            noise = self.wind_turbines[pair["turbine_name"]]["noise_vs_wind_speed"]
            dB_level = self.calculate_sound_level_at_distance(noise, pair["distance"])
            pair["intensity_level_dB"] = dB_level
        return pairs


    def noise_map_at_wind_speeds(self, noise, coord_name, coord_value) -> xr.DataArray:
        """
        Generates a noise map for the wind turbines
        and observation points for each wind speed level.

        :param noise: A 2D array representing the noise level vs wind speed.

        Returns:
            np.ndarray: A 2D array representing the noise map.
        """

        intensity_distance = noise - self.noise_attenuation[..., None]

        # dB at distance
        Z = 10 * np.log10((10 ** (intensity_distance / 10)).sum(axis=2))

        # create xarray to store Z
        Z = xr.DataArray(
            data=Z,
            dims=("lat", "lon", coord_name),
            coords={"lat": self.LAT[:,0], "lon": self.LON[0,:], coord_name: coord_value},
        )

        Z.values = np.clip(Z.values, a_min=0, a_max=None)

        return Z

    def generate_noise_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a noise map for the wind turbines
        and observation points based on the given wind speed.
        """

        # Determine the bounding box for the map

        lat_min = min(
            turbine["position"][0] for turbine in self.wind_turbines.values()
        )
        lat_max = max(
            turbine["position"][0] for turbine in self.wind_turbines.values()
        )
        lon_min = min(
            turbine["position"][1] for turbine in self.wind_turbines.values()
        )
        lon_max = max(
            turbine["position"][1] for turbine in self.wind_turbines.values()
        )
        margin = (1/125)

        # Adjust the map size to include observation points
        # if any are present
        if self.listeners:
            for point in self.individual_noise:
                lat_min = min(lat_min, point["listener_position"][0]) - margin
                lat_max = max(lat_max, point["listener_position"][0]) + margin
                lon_min = min(lon_min, point["listener_position"][1]) - margin
                lon_max = max(lon_max, point["listener_position"][1]) + margin
        else:
            lat_min -= margin
            lat_max += margin
            lon_min -= margin
            lon_max += margin

        lon_array = np.linspace(lon_min, lon_max, NOISE_MAP_RESOLUTION)
        lat_array = np.linspace(lat_min, lat_max, NOISE_MAP_RESOLUTION)
        LON, LAT = np.meshgrid(lon_array, lat_array)

        # Calculate the noise level at each point
        positions = [point["position"] for point in self.wind_turbines.values()]

        distances = np.array(
            [
                haversine(point1=(lat, lon), point2=position, unit=Unit.METERS)
                for lat, lon in zip(LAT.flatten(), LON.flatten())
                for position in positions
            ]
        ).reshape(LAT.shape[0], LAT.shape[1], len(positions))

        noise_attenuation_over_distance = 20 * np.log10(distances)  # dB

        return LAT, LON, noise_attenuation_over_distance

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

            # Plot observation points
            for point, specs in self.listeners.items():
                plt.plot(
                    *specs["position"][::-1], "ro"
                )  # Make sure the position is in (Longitude, Latitude) order
                # add label next to it
                plt.text(
                    specs["position"][1] + 0.002,
                    specs["position"][0] + 0.002,
                    point,
                )

            plt.grid(True)
            plt.show()

    def plot_on_map(self):
        """
        Displays the wind turbines and observation points
        on a real map using their latitude and longitude.
        """

        # Get the average latitude and longitude to center the map
        avg_lat = sum(turbine["position"][0] for turbine in self.wind_turbines.values()) / len(
            self.wind_turbines
        )
        avg_lon = sum(turbine["position"][1] for turbine in self.wind_turbines.values()) / len(
            self.wind_turbines
        )

        # Create a folium map centered at the average latitude and longitude
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

        # Add markers for each wind turbine
        for turbine, specs in self.wind_turbines.items():
            folium.Marker(
                location=specs["position"],
                tooltip=f"{'name'}, {specs['power']} MW Wind Turbine",
                icon=folium.Icon(icon="cloud"),
            ).add_to(m)

        # Add markers for the observation points
        for observation_point, specs in self.listeners.items():
            folium.Marker(
                location=specs["position"],
                tooltip="Observation Point",
                icon=folium.Icon(icon="star", color="red"),
            ).add_to(m)

        # Display the map
        display(m)

