"""
Estimate noise attenuation due to ground type, distance from the source and elevation difference between the source
and receiver, according to ISO 9613-2:2024.
"""

import numpy as np
import requests
from haversine import haversine, Unit



OPEN_ELEVATION_API = "https://api.open-elevation.com/api/v1/lookup?locations="


def calculate_ground_attenuation(distance):
    """
    Calculate the ground attenuation in dB, according to ISO 9613-2:2024.
    """

    elevation_grid = get_elevation_grid(

    )

    # elevated distance is the distance between
    # the source and receiver, considering their
    # respective elevations.


    # h_m in ISO 9613-2:2024 is the mean height of the ground profile
    mean_height = ground_area_profile / elevated_distance
    ground_attenuation = 4.8 - (2 * mean_height / distance) * (17 + (300 / distance))
    return np.where(ground_attenuation < 0, 0, ground_attenuation)