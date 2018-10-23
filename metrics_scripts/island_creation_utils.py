import numpy as np

from shapely.geometry import shape, Point, Polygon, MultiLineString, MultiPoint, MultiPolygon, LineString
from shapely.ops import transform, polygonize_full
import cPickle as pickle

import fiona

from metrics_utils import *



def get_smooth_islands(islands):

    smooth_islands = []

    for n in range(len(islands)):
    
        s = islands[n]

        counter = 0
        while not s.is_valid and (counter < 10):
            s = Polygon(s.exterior.simplify(50))
            counter += 1
        
        smooth_islands.append(s)

    
    smooth_islands = MultiPolygon(smooth_islands)
    
    
    return smooth_islands
    
    

def load_raw_island_shapefile(filename,
                                poly_islands,
                                save_shapefile = True,
                                file_root = ''):



    print 'Load island shapefile'

    islands, DN = load_shapefile(island_filename, ['DN'])
    islands = [islands[n] for n in range(len(islands)) if DN['DN'][n] > 2]



    print 'Finding nested islands'

    islands, contained_islands = find_nested_islands(islands,
                                        merge_islands = True)



    print 'Finding corresponding IDs'

    contained = []

    for m,p in enumerate(poly_islands):
    
        c = [n for n,i in enumerate(islands) if i.within(p)]
    
        if len(c) == 0:
            c = [n for n,i in enumerate(islands)
                    if i.representative_point().within(p)]        
        
        contained.append(c)
    

    contained_multiple = [n for n,i in enumerate(contained) if len(i)>1]
    contained_none = [n for n,i in enumerate(contained) if len(i)==0]




    taken = [c[0] for c in contained if len(c) == 1]

    for p in contained_multiple:
    
        c = [n for n,i in enumerate(islands) if i.within(poly_islands[p])]
        
        largest = c[0]
        
        for n in c:
        
            if not n in taken:
        
                if islands[n].area > islands[largest].area:

                    largest = n
    
        contained[p] = [largest]

    
    
    
    sorted_islands = []

    for i in contained:
        sorted_islands.append(islands[i])

    
    islands = MultiPolygon(sorted_islands)
    
    
    if save_shapefile:
    
        create_shapefile_from_shapely_multi(islands,
                                    file_root + '/islands_sorted.shp')




    return islands




    
    
    
    
def polygonize_shapely_lines(shp, size_threshold = 0):
    
    result, _,_,_ = polygonize_full(shp)
    polys = MultiPolygon([i for i in result if i.area > size_threshold])
    
    return polys
    
    
    
    
def find_nested_islands(poly, merge_islands = True):

    contained_islands = []

    for i in range(len(poly)):

        # if island has an inner ring
        if not poly[i].boundary.equals(poly[i].exterior):

            # check all other islands
            for j in range(len(poly)):
                if i != j:

                    # check if one is inside the other
                    inside = poly[j].within(Polygon(poly[i].exterior))

                    if inside:
                        contained_islands.append([i,j])

    
    if merge_islands:
            
        bad_islands = set([i[1] for i in contained_islands])
        holey_islands = set([i[0] for i in contained_islands])

        new_islands = [Polygon(poly[i].exterior)
                        if i in holey_islands else poly[i]
                        for i in range(len(poly))]
                        
        new_islands = [new_islands[i]
                        for i in range(len(new_islands))
                        if i not in bad_islands]

        poly = MultiPolygon(new_islands)
        
    
    return poly, contained_islands
    
    
    
    
    
def find_bounding_channels(shp,
                            poly,
                            save = True,
                            load_saved = False,
                            file_root = ''):

    if load_saved:
        bounds = pickle.load(open(
                        file_root + '/island_boundary_channels.p', "rb"))
                                  
        interior_channels = pickle.load(open(
                            file_root + '/island_interior_channels.p', "rb"))



    else:

        midpts = [l.interpolate(0.5, normalized=True).buffer(5) for l in shp]

        bounds = []
        interior_channels = []

        # check if line midpoints intersect island outlines
        # to identify which lines make up each island
    
        for polygon in poly:
    
            touch = [i for i,l in enumerate(midpts)
                     if polygon.exterior.intersects(l)]
            bounds.append(touch)
        
            touch = [i for i,l in enumerate(midpts)
                     if polygon.contains(l)]
            interior_channels.append(touch)
            
            
        if save:
        
            pickle.dump(bounds, open(
                        file_root + '/island_boundary_channels.p', "wb"))
                        
            pickle.dump(interior_channels, open(
                        file_root + '/island_interior_channels.p', "wb"))
        
    
    return bounds, interior_channels    
    
    
    
    