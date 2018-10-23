import numpy as np
from shapely.geometry import shape, Point, Polygon, MultiLineString, MultiPoint, MultiPolygon, LineString
from scipy.ndimage import morphology


import cPickle as pickle

from metrics_utils import *


def extract_polygon_props(islands,
                          network_lines,
                          interior_channels):


    interior_channel_lengths = [sum([network_lines[j].length
                                for j in interior_channels[i]]) / 1e3
                                if len(interior_channels[i])>0 else 0
                                for i in range(len(islands))]
    
    
    perimeter = np.array([i.boundary.length for i in islands]) / 1e3
    wetted_perimeter = perimeter + 2 * np.array(interior_channel_lengths)
    
    area = np.array([i.area for i in islands]) / 1e6
    
    perimeter_convex_hull = np.array([i.convex_hull.exterior.length
                                      for i in islands]) / 1e3
    area_convex_hull = np.array([i.convex_hull.area for i in islands]) / 1e6

    a = np.array(map(Polygon_axes, islands))
    minor_axis = a[:,0] / 1e3
    major_axis = a[:,1] / 1e3
    poly_orientation = a[:,2]
    aspect_ratio = major_axis / minor_axis

    circularity = 4 * np.pi * area / perimeter**2
    
    equivalent_area_diameter = np.sqrt((4 / np.pi) * area)
    perimeter_equivalent_diameter = area / np.pi
    
    solidity = area / area_convex_hull
    concavity = area_convex_hull - area
    convexity = perimeter_convex_hull / perimeter
    
    dry_shape_factor = perimeter / np.sqrt(area)
    wet_shape_factor = wetted_perimeter / np.sqrt(area)
    
        
    num_ox = []

    for i in islands:
        oxs = 0
        
        for j in i.interiors:

            try:
                a = Polygon(j).buffer(-20).area

                if a > 40000:
                
                    oxs += 1
            except:
                pass

        num_ox.append(oxs)

    poly_metrics = {'p_area': area,
                    'p_perim': perimeter,
                    'p_wetperim': wetted_perimeter,
                    'p_ch_area': area_convex_hull,
                    'p_ch_perim': perimeter_convex_hull,
                    'p_major_ax': major_axis,
                    'p_minor_ax': minor_axis,
                    'p_asp_rat': aspect_ratio,
                    'p_orient': poly_orientation,
                    'p_circ': circularity,
                    'p_eq_a_dia': equivalent_area_diameter,
                    'p_p_eq_dia': perimeter_equivalent_diameter,
                    'p_solidity': solidity,
                    'p_concav': concavity,
                    'p_convex': convexity,
                    'p_d_shapef': dry_shape_factor,
                    'p_w_shapef': wet_shape_factor,
                    'p_num_ox': num_ox,
                    'p_int_len': interior_channel_lengths}    
    
    return poly_metrics




def calculate_edge_distances(islands,
                             save = True,
                             load_saved = False,
                             file_root = ''):

    if load_saved:

        edgedists = pickle.load( open( file_root + '/edge_distances.p', "rb" ))

    else:

        edgedists = np.zeros((len(islands),))

        for n,s in enumerate(islands):

            cellsize = 50
            minx, miny, maxx, maxy = s.envelope.bounds

            if (((maxx - minx) / cellsize > 1000) or
                   ((maxy - miny) / cellsize > 1000)):

                cellsize = 100
                
            if (((maxx - minx) / cellsize > 5000) or
                   ((maxy - miny) / cellsize > 5000)):

                cellsize = 500


            minx = np.floor(minx) - 1 * cellsize
            maxx = np.ceil(maxx) + 2 * cellsize
            miny = np.floor(miny) - 1 * cellsize
            maxy = np.ceil(maxy) + 2 * cellsize

            x = np.arange(minx, maxx , cellsize)
            y = np.arange(miny, maxy , cellsize)


            mask = outline_to_mask(s.exterior, x, y)
            distmap = morphology.distance_transform_edt(mask)

            edgedists[n] = distmap.max() * cellsize

        if save:

            pickle.dump(edgedists,
                open( file_root + '/edge_distances' + '.p', "wb" ) )

    return edgedists







