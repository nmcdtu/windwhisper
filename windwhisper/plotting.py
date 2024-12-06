import numpy as np
import matplotlib.pyplot as plt
import folium
from folium import Map, Element
from pyproj import Transformer
import geojson
from shapely.geometry import LineString, mapping

color_map = {
        10: "blue",
        20: "darkblue",
        30: "green",
        40: "yellow",
        50: "orange",
        60: "red",
        70: "purple",
    }


def generate_contours(data, levels, name="Noise Contours"):
    """
    Generate contours using Matplotlib and create a GeoJSON object for Folium.
    """

    # Generate the contours
    c = plt.contour(
        data.lon, data.lat, data.data,  # Transpose the data to match Matplotlib's expectation
        levels=levels,
        colors=[color_map[k] for k in color_map.keys()],
        linewidths=1.5
    )

    # Create a list to store GeoJSON features
    geojson_features = []

    for i, collection in enumerate(c.collections):
        for path in collection.get_paths():
            for line in path.to_polygons():
                # Transform projected coordinates to EPSG:4326
                coords = [
                    (x, y) for x, y in line
                ]
                # Add the geometry as a GeoJSON feature
                geojson_features.append(
                    geojson.Feature(
                        geometry=mapping(LineString(coords)),
                        properties={"level": c.levels[i], "color": color_map[c.levels[i]]}
                    )
                )

    # Create a GeoJSON FeatureCollection
    geojson_object = geojson.FeatureCollection(geojson_features)

    # Add GeoJSON to a Folium FeatureGroup
    feature_group = folium.FeatureGroup(name=name)
    folium.GeoJson(
        geojson_object,
        style_function=lambda feature: {
            "color": feature["properties"]["color"],
            "weight": 2,
        }
    ).add_to(feature_group)

    return feature_group


def add_legend(map_object):
    """
    Adds a legend to the Folium map.

    :param map_object: The Folium map object to add the legend to.
    :param color_map: A dictionary mapping contour levels to colors.
    """
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 50px; left: 15px; width: 200px; height: auto; 
        background-color: white; z-index:9999; font-size:14px; 
        border:2px solid grey; padding: 10px; border-radius: 5px;
    ">
    <strong>Legend</strong><br>
    """
    for level, color in color_map.items():
        legend_html += f"""
        <i style="background: {color}; width: 10px; height: 10px; display: inline-block; margin-right: 10px;"></i>
        {level} dB(A)<br>
        """
    legend_html += "</div>"

    # Wrap the HTML in a folium Element and add it to the map
    legend = Element(legend_html)
    map_object.get_root().html.add_child(legend)

    return map_object



def generate_map(noise_dataset):

    # Create Folium map
    center_lat = np.mean(noise_dataset.lat.values)
    center_lon = np.mean(noise_dataset.lon.values)

    m = folium.Map(location=[center_lat, center_lon])


    # # Add the contours to the map
    contours = generate_contours(
        noise_dataset["combined"],
        levels=[30, 40, 50, 60, 70],
        name="Overall noise levels"
    )
    m.add_child(contours)
    # #
    contours = generate_contours(
        noise_dataset["wind"],
        levels=[30, 40, 50, 60, 70],
        name="Wind turbine(s) noise levels"
    )
    m.add_child(contours)
    # #
    contours = generate_contours(
        noise_dataset["ambient"],
        levels=[30, 40, 50, 60, 70],
        name="Pre-existing noise levels"
    )
    m.add_child(contours)
    # #
    contours = generate_contours(
        noise_dataset["net"],
        levels=[0, 20, 30, 40],
        name="Net contribution"
    )
    m.add_child(contours)

    contours = generate_contours(
        noise_dataset["flip"],
        levels=[50, 60, 70],
        name="Area beyond WHO guidelines"
    )
    m.add_child(contours)

    # Add the legend
    m = add_legend(m)

    # Add layer control for toggling
    folium.LayerControl().add_to(m)

    # Save the map as an HTML file
    m.save("noise_map.html")

