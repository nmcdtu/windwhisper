"""
Estimate noise attenuation due to ground type, distance from the source and elevation difference between the source
and receiver, according to ISO 9613-2:2024.
"""

import numpy as np
import requests
from haversine import haversine, Unit

OPEN_ELEVATION_API = "https://api.open-elevation.com/api/v1/lookup?locations="

def get_elevation_along_path(source_coordinates, receive_coordinates):
    """
    Calculate the elevation difference between the source and receiver.

    :param source_coordinates: Coordinates of the source, using EPSG:4326.
    :param receive_coordinates: Coordinates of the receiver, using EPSG:4326.
    """

    # determine distance, in meters, between source and receiver
    distance = haversine(source_coordinates, receive_coordinates, unit=Unit.METERS)

    start_lon, start_lat = source_coordinates
    end_lon, end_lat = receive_coordinates

    # create path between source and receiver: one point every 10 meters
    path = [(start_lon + i * (end_lon - start_lon) / distance, start_lat + i * (end_lat - start_lat) / distance)
            for i in range(0, int(distance), 10)]


    # get elevation data from Open Elevation API
    url = OPEN_ELEVATION_API
    elevations = []
    for lon, lat in path:
        url += f"{lat},{lon}|"

    url = url[:-1]

    response = requests.get(url)

    if response.status_code == 200:
        elevations = response.json()["results"]
    else:
        print(f"Failed to fetch elevation data: {response.status_code}")

def calculate_





def calculate_ground_attenuation(distance):
    """
    Calculate the ground attenuation in dB, according to ISO 9613-2:2024.
    """


    # elevated distance is the distance between
    # the source and receiver, considering their
    # respective elevations.


    # h_m in ISO 9613-2:2024 is the mean height of the ground profile
    mean_height = ground_area_profile / elevated_distance
    ground_attenuation = 4.8 - (2 * mean_height / distance) * (17 + (300 / distance))
    return np.where(ground_attenuation < 0, 0, ground_attenuation)