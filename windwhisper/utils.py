from pyproj import Transformer

def create_bounding_box(center_x: float, center_y: float, buffer_meters: int) -> tuple:
    """
    Creates a bounding box around a center point with a specified buffer.

    Parameters:
        center_x (float): X coordinate of the center point.
        center_y (float): Y coordinate of the center point.
        buffer_meters (float): Buffer distance in meters.

    Returns:
        tuple: Bounding box as (min_x, min_y, max_x, max_y).
    """
    min_x = center_x - buffer_meters
    max_x = center_x + buffer_meters
    min_y = center_y - buffer_meters
    max_y = center_y + buffer_meters

    return min_x, min_y, max_x, max_y


def translate_4326_to_3035(lon: float, lat: float) -> tuple:
    # Initialize the transformer for EPSG:4326 to EPSG:3035
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)

    return transformer.transform(lon, lat)
