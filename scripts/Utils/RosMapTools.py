from nav_msgs.msg import OccupancyGrid
import numpy as np
import math


def get_area(map, position, area, threshold = 10):
    # position in "map" frame
    size = [map.map.info.width, map.map.info.height]
    npmap = np.asarray(map.map.data).reshape(size)
    resolution = map.map.info.resolution
    area_in_pixels = [round(area/resolution), round(area/resolution)]
    position_in_pixel = [round(position[0]/resolution),
                         round(position[1]/resolution)]
    map_area = np.zeros(area_in_pixels, dtype="uint8")
    # TODO: rewrite this with np methods
    mapx1 = max((position_in_pixel[0]-round(area_in_pixels[0]/2)), 0)
    mapx2 = min((position_in_pixel[0]+round(area_in_pixels[0]/2)), size[0])
    mapy1 = max((position_in_pixel[1]-round(area_in_pixels[1]/2)), 0)
    mapy2 = min((position_in_pixel[1]+round(area_in_pixels[1]/2)), size[1])
    areax1 = -min(position_in_pixel[0] - round(area_in_pixels[0]/2), 0)
    areax2 = areax1 + mapx2 - mapx1
    areay1 = -min(position_in_pixel[1] - round(area_in_pixels[1]/2), 0)
    areay2 = areay1 + mapy2 - mapy1
    map_area[areax1:areax2, areay1:areay2] = npmap[mapx1:mapx2, mapy1:mapy2]

    # prepare data
    min_obstacle_val = 10
    threshold = map_area < min_obstacle_val
    map_area[threshold] = 0
    map_area[~threshold] = 1
    return map_area, resolution
