import numpy as np
from scipy.ndimage import morphology
from scipy import signal
import skimage
from scipy import ndimage


def island_properties(islands, smooth = True, properties = True):
    '''
    Identifies islands and calculates island morphological properties
    
    Input:
    -------
    mapfile: dictionary containing, at least,
             a mask of islands ('islandmap')
             and a mask of land ('landmap')
    smooth: boolean for median filter to simplify island mask
    properties: boolean to calculate properties
    
    Output:
    --------
    A list containing:
    island_IMG: array of islands by label
    
    If properties = True, also contains:
    island_props: dictionary of island properties, by label
    ecdf: cumulative probability distributions of island size
    edgedist_IMG: array of islands with calculated distances from edges
    edgedist_hist: histogram of edge distances
    
    '''

    tot_area = islands.sum()

    if smooth: 
        islands_filt = signal.medfilt2d(islands.astype(float), 3)
    else:
        islands_filt = islands.astype(float)

    islandmap, N = ndimage.label(islands_filt)
    
    
    
    island_props = {}
    
    # for island properties
    rps = skimage.measure.regionprops(islandmap, cache=False)

    Li = [r.major_axis_length for r in rps if r.minor_axis_length > 0]
    Wi = [r.minor_axis_length for r in rps if r.minor_axis_length > 0]
    Pi = [r.perimeter for r in rps if r.minor_axis_length > 0]
    Ai = [r.area for r in rps if r.minor_axis_length > 0]
    label = [r.label for r in rps if r.minor_axis_length > 0]

    # island_area = [float(a) / tot_area for a in Ai]
    island_area = [float(a) / tot_area for a in Ai]
    island_aspect_ratio = [Li[n] / Wi[n] for n in range(len(Li))]
    island_shape_factor = [Pi[n] / np.sqrt(Ai[n]) for n in range(len(Li))]


    island_area = np.array(island_area)

    
    island_props['major_axis'] = Li
    island_props['minor_axis'] = Wi
    island_props['perimeter'] = Pi
    island_props['area'] = island_area
    island_props['aspt_ratio'] = island_aspect_ratio
    island_props['edge_dist'] = island_edge_dist
    island_props['shp_factor'] = island_shape_factor
    island_props['label'] = label
    
    returnlist = [islandmap, island_props]
    
    
    
    EdgeDistMap = None
    histogram = None
    ecdf_results = None
    
    if properties:
    
        EdgeDistMap = morphology.distance_transform_edt(islands_filt)
        
        # for edge distance map
        bins = np.arange(0, np.ceil(EdgeDistMap.max() + 1))
        bin_centers = bins[1:]

        count, bins = np.histogram(EdgeDistMap, bins + 0.5)

        histogram = {}
        histogram['count'] = count
        histogram['bin_centers'] = bin_centers
        

        island_edge_dist = [EdgeDistMap[islandmap == n].max() for n in range(1,N+1) if rps[n-1].minor_axis_length > 0]

        area_min = 1e-5
        quantiles, cumprob = ecdf(island_area[island_area > area_min])
        
        island_props['edge_dist'] = island_edge_dist
        
        ecdf_results = [quantiles, cumprob]
        
        
        returnlist = [islandmap, island_props, ecdf_results, EdgeDistMap, histogram]


    return returnlist
    

    
def ecdf(sample):
    '''
    Cumulative frequency distribution of island sizes
    
    Called by island_properties()
    '''

    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob    


def fractal_dimension(data):
    '''
    Fractal dimension of image
    
    Input:
    -------
    data: Numpy array. For analyzing DeltaRCM output, data should be
          mapfile['centerlinemap']
          
    Output:
    -------
    D_frac: fractal dimension of image
    '''

    M,N = data.shape

    k = 1
    bsize = 2**(k - 1) # start with box size=1 pixel

    boxsize = []
    boxcount = []

    # starts with the smallest box size (1 pixel)
    # increase box size by 2x each time
    while bsize < min(M,N)/2.:

        boxsize.append(bsize)
        boxcount_k = 0

        for imin in range(1, M-bsize+1, bsize):

            for jmin in range(1, N-bsize+1, bsize):

                imax = imin + bsize - 1
                jmax = jmin + bsize - 1

                # counting boxes that contains value>0 cells
                # change this condition for other type of feature extraction

                box = data[imin-1:imax, jmin-1:jmax]

                if box.sum() > 0:
                    boxcount_k += 1

        boxcount.append(boxcount_k)

        k += 1
        bsize = 2**(k - 1)

    fit = np.polyfit(np.log(1./np.array(boxsize)), np.log(boxcount), 1)
    D_frac = fit[0]

    return D_frac
    
    
    
    
def nearest_edge_distance(mapfile):
    '''
    For a mask of islands, calculates the nearest distance between every
    island pixel and the island edge
    
    Called by island_properties()
    
    Input:
    ------
    mapfile: dictionary containing, at minimum, a mask of islands ('islandmap')
    
    Output:
    -------
    EdgeDistMap: array of the minimum distance between every point inside
                 an island and its edge
    histogram: distribution of edge distances
    '''

    islands = mapfile['islandmap']
    
    EdgeDistMap = morphology.distance_transform_edt(islands)

    bins = np.arange(0, np.ceil(EdgeDistMap.max() + 1))
    bin_centers = bins[1:]

    count, bins = np.histogram(EdgeDistMap, bins + 0.5)

    histogram = {}
    histogram['count'] = count
    histogram['bin_centers'] = bin_centers
    
    return EdgeDistMap, histogram
    
    
    
def fractional_areas(mapfile):
    '''
    Calculates areas (in pixels) and fractional areas of different
    "landcover" categories
    
    Input:
    ------
    mapfile: dictionary of land types, created by function make_map()
    
    Output:
    -------
    areas: dictionary containing values of areas and fractional areas
    '''

    area = {}

    area['land'] = mapfile['landmap'].sum()
    area['shore'] = mapfile['shoremap'].sum()
    area['wet'] = mapfile['allwetmap'].sum()
    area['channel'] = mapfile['channelmap'].sum()

    area['frac_wet'] = area['wet'] / area['land']
    area['frac_channel'] = area['channel'] / area['land']
    area['length_wet'] = mapfile['edgemap'].sum()
    
    return area