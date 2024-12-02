"""
This module models the geometric divergence of sound waves.
"""

import numpy as np


def get_geometric_spread_loss(distance: np.array) -> float:
    """
    Calculate the geometric spread loss in dB, according to ISO 9613-2:2024.
    :param distance: The distance between the source and receiver in meters.
    :return: The geometric spread loss

    """
    return 20 * np.log10(distance / 1) + 11
