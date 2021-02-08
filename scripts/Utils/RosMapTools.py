from nav_msgs.msg import OccupancyGrid
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
# plt.imshow(bool_grid_map, cmap='hot', interpolation='nearest')
# plt.show()

def get_coordinate_on_map(point, map, map_origin = np.array([0.,0.])):
    resolution = map.map.info.resolution
    position = point+map_origin
    position_in_pixel = np.array([round(position[1]/resolution),round(position[0]/resolution)])
    size = np.array([map.map.info.height,map.map.info.width]) # TODO: check map mirroring throw axes
    
    if (position_in_pixel < 0).any() or (position_in_pixel > size).any():
        position_in_pixel == None
    return position_in_pixel

def map_to_bool_grid_map(map, threshold = 10):
    # position in "map" frame
    size = [map.map.info.height,map.map.info.width] # TODO: check map mirroring throw axes
    # size = [map.map.info.width,map.map.info.height] # TODO: check map mirroring throw axes
    npmap = np.asarray(map.map.data).reshape(size)
    # prepare data
    bool_grid_map = npmap < threshold
    # plt.imshow(bool_grid_map, cmap='hot', interpolation='nearest')
    # plt.show()
    # grid_map[threshold_map] = 0
    # grid_map[~threshold_map] = 1
    # grid_map = map_area[::-1, ...]
    # return grid_map
    return bool_grid_map

def get_areas(map, positions, area, threshold = 10):
    # position in "map" frame
    size = [map.map.info.height,map.map.info.width] # TODO: check map mirroring throw axes
    # size = [map.map.info.width,map.map.info.height] # TODO: check map mirroring throw axes
    npmap = np.asarray(map.map.data).reshape(size)
    # npmap = np.asarray(map.map.data).reshape(size)
    # plt.imshow(npmap, cmap='hot', interpolation='nearest')
    # plt.show()
    resolution = map.map.info.resolution
    area_in_pixels = [round(area/resolution), round(area/resolution)]
    maps_area = []
    for map_origin in positions:
        position_in_pixel = [round(map_origin[1]/resolution),round(map_origin[0]/resolution)]
        # position_in_pixel = [round(position[0]/resolution),round(position[1]/resolution)]
        map_area = np.zeros(area_in_pixels, dtype="uint8")
        # TODO: rewrite this with np methods
        mapx1 = min(max((position_in_pixel[0]-round(area_in_pixels[0]/2)), 0), size[0])
        mapx2 = max(min((position_in_pixel[0]+round(area_in_pixels[0]/2)), size[0]),0)
        mapy1 = min(max((position_in_pixel[1]-round(area_in_pixels[1]/2)), 0),size[1])
        mapy2 = max(min((position_in_pixel[1]+round(area_in_pixels[1]/2)), size[1]),0)
        areax1 = -min(position_in_pixel[0] - round(area_in_pixels[0]/2), 0)
        areax2 = areax1 + mapx2 - mapx1
        areay1 = -min(position_in_pixel[1] - round(area_in_pixels[1]/2), 0)
        areay2 = areay1 + mapy2 - mapy1
        map_area[areax1:areax2, areay1:areay2] = npmap[mapx1:mapx2, mapy1:mapy2]

        # prepare data
        threshold_map = map_area < threshold
        map_area[threshold_map] = 0
        map_area[~threshold_map] = 1
        map_area = map_area[::-1, ...]
        maps_area.append(map_area)
    # cutting area debug info
    # if (map_area>0).any():
    #     plt.imshow(map_area, cmap='hot', interpolation='nearest')
    #     plt.show()
    return maps_area, resolution




def get_area(map, position, area, threshold = 10):
    # position in "map" frame
    size = [map.map.info.height,map.map.info.width] # TODO: check map mirroring throw axes
    # size = [map.map.info.width,map.map.info.height] # TODO: check map mirroring throw axes
    npmap = np.asarray(map.map.data).reshape(size)
    # npmap = np.asarray(map.map.data).reshape(size)
    # plt.imshow(npmap, cmap='hot', interpolation='nearest')
    # plt.show()
    resolution = map.map.info.resolution
    area_in_pixels = [round(area/resolution), round(area/resolution)]
    position_in_pixel = [round(position[1]/resolution),round(position[0]/resolution)]
    # position_in_pixel = [round(position[0]/resolution),round(position[1]/resolution)]
    map_area = np.zeros(area_in_pixels, dtype="uint8")
    # TODO: rewrite this with np methods
    mapx1 = min(max((position_in_pixel[0]-round(area_in_pixels[0]/2)), 0), size[0])
    mapx2 = max(min((position_in_pixel[0]+round(area_in_pixels[0]/2)), size[0]),0)
    mapy1 = min(max((position_in_pixel[1]-round(area_in_pixels[1]/2)), 0),size[1])
    mapy2 = max(min((position_in_pixel[1]+round(area_in_pixels[1]/2)), size[1]),0)
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
    map_area = map_area[::-1, ...]
    # cutting area debug info
    # if (map_area>0).any():
    #     plt.imshow(map_area, cmap='hot', interpolation='nearest')
    #     plt.show()
    return map_area, resolution
