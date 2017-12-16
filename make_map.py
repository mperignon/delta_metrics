import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from skimage import feature

from Seaangles_mod import Seaangles_mod
from ModifiedSingularityIndex2D import ModifiedSingularityIndex2D

def make_map(topo,
             vel,
             topo_threshold = -0.5,
             angle_threshold = 75,
             velocity_threshold = 0.3,
             numviews = 3,
             save_file = True,
             sroot = './',
             fname = 'mapfile'):


    oceanmap = (topo < topo_threshold) * 1.

    shoreangles, waves1s, seaangles, picshore = Seaangles_mod(numviews, oceanmap)



    shore_image = np.zeros_like(topo)
    flat_indices = map(lambda x: np.ravel_multi_index(x, shore_image.shape),
                                     seaangles[:2,:].T.astype(int))
    shore_image.flat[flat_indices] = seaangles[-1,:]



    cs = plt.contour(shore_image, [angle_threshold])
    plt.close()
    C = cs.allsegs[0][0]

    shoremap = np.zeros_like(topo)
    flat_indices = map(lambda x: np.ravel_multi_index(x, shoremap.shape),
                                     np.fliplr(np.round(C).astype(int)))
    shoremap.flat[flat_indices] = 1


    landmap = (shore_image < angle_threshold) * 1
    wetmap = oceanmap * landmap
    flowmap = (vel > velocity_threshold) * 1
    channelmap = np.minimum(landmap, flowmap)

    min_scale = 1.1
    nrScales = 3
    SI = ModifiedSingularityIndex2D(channelmap, nrScales, 1, min_scale)

    centerlinemap = SI['posNMS'] > 0.01

    mapfile = {}
    mapfile['wetmap'] = wetmap
    mapfile['landmap'] = landmap
    mapfile['shoremap'] = shoremap
    mapfile['channelmap'] = channelmap
    mapfile['centerlinemap'] = centerlinemap
    
    mapfile['allwetmap'] = ((mapfile['wetmap'] +
                             mapfile['channelmap']) > 0) * 1.
                             
    mapfile['edgemap'] = np.maximum(0,
                                    feature.canny(mapfile['allwetmap'])*1 -
                                    feature.canny(mapfile['landmap'])*1)


    if save_file:
        pickle.dump( mapfile, open( sroot + fname + '.p', "wb" ) )


    return mapfile