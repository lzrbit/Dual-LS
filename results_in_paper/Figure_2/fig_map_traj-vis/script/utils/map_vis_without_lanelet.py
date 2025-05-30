

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt

import xml.etree.ElementTree as xml
import pyproj
import math

from .dict_utils import *


class Point:
    def __init__(self):
        self.x = None
        self.y = None


class LL2XYProjector:
    def __init__(self, lat_origin, lon_origin):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = math.floor((lon_origin + 180.) / 6) + 1  # works for most tiles, and for all in the dataset
        self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat, lon):
        [x, y] = self.p(lon, lat)
        return [x - self.x_origin, y - self.y_origin]


def get_type(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "type":
            return tag.get("v")
    return None


def get_subtype(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "subtype":
            return tag.get("v")
    return None


def get_x_y_lists(element, point_dict):
    x_list = list()
    y_list = list()
    for nd in element.findall("nd"):
        pt_id = int(nd.get("ref"))
        point = point_dict[pt_id]
        x_list.append(point.x)
        y_list.append(point.y)
    return x_list, y_list


def set_visible_area(point_dict, axes):
    min_x = 10e9
    min_y = 10e9
    max_x = -10e9
    max_y = -10e9

    for id, point in get_item_iterator(point_dict):
        min_x = min(point.x, min_x)
        min_y = min(point.y, min_y)
        max_x = max(point.x, max_x)
        max_y = max(point.y, max_y)

    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([min_x - 1, max_x + 1])
    axes.set_ylim([min_y - 1, max_y +1])   # + 1


def draw_map_without_lanelet(filename, axes, lat_origin, lon_origin, alpha):
    assert isinstance(axes, matplotlib.axes.Axes)

    axes.set_aspect('equal', adjustable='box')
    axes.patch.set_facecolor('white')

    projector = LL2XYProjector(lat_origin, lon_origin)

    e = xml.parse(filename).getroot()

    point_dict = dict()
    for node in e.findall("node"):
        point = Point()
        point.x, point.y = projector.latlon2xy(float(node.get('lat')), float(node.get('lon')))
        point_dict[int(node.get('id'))] = point

    set_visible_area(point_dict, axes)

    unknown_linestring_types = list()
    
    # Initialize a common type_dict to include alpha
    type_dict = {'color': 'black', 'linewidth': 1, 'zorder': 10, 'alpha': alpha}

    for way in e.findall('way'):
        way_type = get_type(way)
        if way_type is None:
            raise RuntimeError("Linestring type must be specified")
        
        # Reset type_dict for each way, maintaining the alpha value
        if way_type == "curbstone":
            type_dict = {'color': "black", 'linewidth': 1, 'zorder': 10, 'alpha': alpha}
        elif way_type in ["line_thin", "line_thick"]:
            way_subtype = get_subtype(way)
            linewidth = 1 if way_type == "line_thin" else 2
            if way_subtype == "dashed":
                type_dict = {'color': "white", 'linewidth': linewidth, 'zorder': 10, 'dashes': [10, 10], 'alpha': alpha}
            else:
                type_dict = {'color': "white", 'linewidth': linewidth, 'zorder': 10, 'alpha': alpha}
        elif way_type in ["pedestrian_marking", "bike_marking", "stop_line"]:
            dashes = [5, 10] if way_type in ["pedestrian_marking", "bike_marking"] else []
            linewidth = 1 if way_type in ["pedestrian_marking", "bike_marking"] else 3
            type_dict = {'color': "white", 'linewidth': linewidth, 'zorder': 10, 'dashes': dashes, 'alpha': alpha}
        elif way_type == "virtual":
            type_dict = {'color': "blue", 'linewidth': 1, 'zorder': 10, 'dashes': [2, 5], 'alpha': alpha}
        elif way_type in ["road_border", "guard_rail"]:
            type_dict = {'color': "black", 'linewidth': 1, 'zorder': 10, 'alpha': alpha}
        elif way_type == "traffic_sign":
            continue
        else:
            if way_type not in unknown_linestring_types:
                unknown_linestring_types.append(way_type)
            continue

        x_list, y_list = get_x_y_lists(way, point_dict)
        axes.plot(x_list, y_list, **type_dict)

    if len(unknown_linestring_types) != 0:
        print("Found the following unknown types, did not plot them: " + str(unknown_linestring_types))
