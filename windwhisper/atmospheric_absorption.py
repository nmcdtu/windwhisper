"""
This module contains functions to calculate the atmospheric absorption of sound.
"""

import numpy as np
from . import DATA_DIR
import yaml


def load_atmospheric_absorption_coefficients():
    """
    Load YAML file containing the atmospheric absorption coefficients.
    """

    with open(DATA_DIR / "absorption_coefficients.yaml") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def compute_weighted_absorption(absorption_coefficients):
    """
    Compute the weighted absorption coefficient directly using spectral levels in dB.

    :param absorption_coefficients: Dictionary of absorption coefficients (dB/km) by frequency band (Hz).
    :return: Weighted absorption coefficient (alpha_weighted) in dB/km.
    """
    # Third-octave bands and their A-weighting corrections
    frequencies = list(absorption_coefficients.keys())
    a_weighting = [-26.2, -16.1, -8.6, -3.2, 0, 1.2, 1.0, -1.1]  # ISO standard corrections

    spectral_levels = [
        85,  # 63 Hz
        90,  # 125 Hz
        96,  # 250 Hz
        97,  # 500 Hz
        100,  # 1000 Hz
        99,  # 2000 Hz
        97,  # 4000 Hz
        96,  # 8000 Hz
    ]


    # Compute unweighted levels by reversing A-weighting corrections
    unweighted_band_levels = [level - aw for level, aw in zip(spectral_levels, a_weighting)]

    # Compute energy contributions for each band
    energy_contributions = [10 ** (level / 10) for level in unweighted_band_levels]

    # Calculate the weighted absorption coefficient
    alpha_weighted = sum(
        energy * absorption_coefficients[freq]
        for energy, freq in zip(energy_contributions, frequencies)
    ) / sum(energy_contributions)

    return alpha_weighted


def get_absorption_coefficient(temperature=20, humidity=70):
    """
    Get the atmospheric absorption coefficient for a given noise level, temperature, and humidity.

    :param temperature: The temperature in degrees Celsius. Default is 20.
    :type temperature: float
    :param humidity: The relative humidity in percent. Default is 70.
    :type humidity: float
    :return: The atmospheric absorption coefficient, in dB/km.

    """

    # Load the atmospheric absorption coefficients
    coefficients = load_atmospheric_absorption_coefficients()

    if temperature not in coefficients:
        raise ValueError(f"Temperature {temperature} not found in the absorption coefficients. "
                         f"Try one of {list(coefficients.keys())}")

    else:
        if humidity not in coefficients[temperature]:
            raise ValueError(f"Humidity {humidity} not found in the absorption coefficients. "
                             f"Try one of {list(coefficients[temperature].keys())}")
        else:
            coefficients = coefficients[temperature][humidity]
            return compute_weighted_absorption(coefficients)
