import numpy as np
from scipy.ndimage import morphology
from scipy import signal
import skimage


def island_properties(mapfile):

    islands = mapfile['islandmap']
    landmap = mapfile['landmap']

    islands_filt = signal.medfilt2d(islands.astype(float), 3)

    islandmap, N = skimage.measure.label(islands_filt, return_num = True)
    EdgeDistMap = morphology.distance_transform_edt(islands_filt)

    rps = skimage.measure.regionprops(islandmap, cache=False)

    Li = [r.major_axis_length for r in rps if r.area > 1]
    Wi = [r.minor_axis_length for r in rps if r.area > 1]
    Pi = [r.perimeter for r in rps if r.area > 1]
    Ai = [r.area for r in rps if r.area > 1]

    island_area = [float(a) / landmap.sum() for a in Ai]
    island_aspect_ratio = [Li[n] / Wi[n] for n in range(len(Li))]
    island_shape_factor = [Pi[n] / np.sqrt(Ai[n]) for n in range(len(Li))]
    island_edge_dist = [EdgeDistMap[islandmap == n].max() for n in range(1,N+1) if rps[n-1].area > 1]

    island_area = np.array(island_area)
    area_min = 1e-5
    quantiles, cumprob = ecdf(island_area[island_area > area_min])

    island_properties = {}

    island_properties['ecdf'] = [quantiles, cumprob]
    island_properties['island_area'] = island_area
    island_properties['island_aspect_ratio'] = island_aspect_ratio
    island_properties['island_edge_dist'] = island_edge_dist
    island_properties['island_shape_factor'] = island_shape_factor

    return island_properties
    
    
def ecdf(sample):

    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)

    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob    


def fractal_dimension(data):

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

#     islands = np.minimum(1, mapfile['wetmap'] + (1 - mapfile['landmap'])) == 0
    
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

    area = {}

    area['land'] = mapfile['landmap'].sum()
    area['shore'] = mapfile['shoremap'].sum()
    area['wet'] = mapfile['allwetmap'].sum()
    area['channel'] = mapfile['channelmap'].sum()

    area['frac_wet'] = area['wet'] / area['land']
    area['frac_channel'] = area['channel'] / area['land']
    area['length_wet'] = mapfile['edgemap'].sum()
    
    return area