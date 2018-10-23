import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from ..metrics import *
from ..spatial_dist import *




from shapely.geometry import shape, Point, Polygon, MultiLineString, MultiPoint, MultiPolygon, LineString

import rasterio
from rasterio.features import shapes

from osgeo import ogr, gdal, osr
from shapely.ops import transform, polygonize_full
from scipy.ndimage import morphology

import cPickle as pickle
import fiona
import itertools
import clusterpy
from scipy.cluster.vq import kmeans,vq
from numpy.random import rand
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import networkx as nx
import matplotlib as mpl


from metrics_util import *
from island_creation_utils import *



mask_filename = '../ganges_metrics/gangeschan.tif'
downsample = 1

network_filename = '../ganges_metrics/network.shp'
channel_parameters = ['Width']
size_threshold = 800000

output_dir = 'Output'




# load water mask raster
island_mask, params = read_tiff_as_array(mask_filename,
                                        get_info = True,
                                        normalize = True)
                                        
r_xmax, r_xmin, r_ymax, r_ymin, r_xres, r_yres = params
pixel_size = r_xres

# set the size_threshold to what we've been using
size_threshold = pixel_size * pixel_size * 10

# reduce the size of the domain, if desired
if downsample > 1:
    island_mask = island_mask[::downsample,::downsample]
    pixel_size = r_xres * downsample

# create a mapfile for later use
mapfile = {}
mapfile['landmap'] = island_mask
mapfile['islandmap'] = island_mask



# load network shapefile
network_lines, network_params = load_shapefile(network_filename,
                                               channel_parameters)


# turn network into polygons
islands = polygonize_shapely_lines(network_lines,
                                   size_threshold = size_threshold)
                                   
      
# find nested islands, merge them                         
islands, nested_islands = find_nested_islands(islands,
                                              merge_islands = True)
                                                   

# find bounding and interior channels for each island                                                   
bounds, interior_channels = find_bounding_channels(network_lines,
                                           islands,
                                           load_saved = True,
                                           file_root = '../ganges_metrics')

flat_bounds = np.unique([item for sublist in bounds for item in sublist])
channel_bounds = MultiLineString([network_lines[i] for i in flat_bounds])



# save islands as shapefile and raster
create_shapefile_from_shapely_multi(channel_bounds,
                                    output_dir + '/islands_polygons.shp',
                                    buffer_width = 150)


create_tiff_from_shapefile(source = output_dir + '/islands_polygons.shp',
                   target = output_dir + '/islands_polygons.tif',
                   reference_raster = '../ganges_metrics/gangeschan.tif')



# calculate average and maximum channel widths for all islands
avg_width, max_width = extract_channel_props(network_lines,
                                             network_params['Width'],
                                             bounds)




























