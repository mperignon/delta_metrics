import numpy as np
import cPickle as pickle

def channel_overlap(filenames):
    '''
    Accepts a list of filenames for mapfiles and
    compares the position of their channels.
    
    Follows same order as the list
    '''

    # first mapfile in list
    f = filenames[0]
    mapfile = pickle.load( open( f, "rb" ) )
    data = mapfile['channelmap']

    # create empty overlap map arrays
    overlapmap = np.zeros(data.shape)
    difference, phi, O_phi = np.zeros((3, len(filenames,)))

    # create circular mask
    maskmap, area_mask = create_circular_mask(data, data.shape[1]/2., 0, 55, 5)

    # get base image values
    chan_base = data * maskmap
    fw_base = chan_base.sum() / area_mask


    for n,f in enumerate(filenames):

        mapfile = pickle.load( open( f, "rb" ) )
        data = mapfile['channelmap']

        chan_step = data * maskmap
        fw_step = chan_step.sum() / area_mask

        difference[n] = np.abs(chan_base - chan_step).sum()

        phi[n] = fw_base * (1 - fw_step) + (1 - fw_base) * fw_step
        O_phi[n] = 1 - difference[n] / (area_mask * phi[n])

        overlapmap = overlapmap + chan_step


    return overlapmap, difference, phi, O_phi
    
    
    
def create_circular_mask(data, x0, y0, r1, r2):
    
    n = max(data.shape)
    y,x = np.ogrid[-y0:n-y0, -x0:n-x0]

    mask = (x*x + y*y <= r1*r1) & (x*x + y*y >= r2*r2)

    array = np.zeros((n, n))
    array[mask] = 1
    maskmap = array[:data.shape[0], :data.shape[1]] > 0
    area_mask = maskmap.sum()
    
    return maskmap, float(area_mask)