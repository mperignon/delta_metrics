import numpy as np
from scipy.ndimage import morphology


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

    islands = np.minimum(1, mapfile['wetmap'] + (1 - mapfile['landmap'])) == 0
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