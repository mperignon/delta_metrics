from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from file_util import load_matlab_data

def water_RGB_image(eta, uw, qw):
    '''
    Creates an RGBA numpy array highlighting the areas of flow
    
    Input:
    --------
    eta: elevation array (i x j)
    uw: flow velocity array (i x j)
    qw: discharge array (i x j)
    
    Output:
    --------
    IMG: (i x j x 4) array
    
    '''
    
    
    IMG = np.zeros((uw.shape + (4,)))

    etaMIN = -8
    etaMAX = 2
    eta_index = np.maximum(1, (255 * (eta-etaMIN)/(etaMAX-etaMIN)).astype(int))

    RGBA = np.array([list(cm.gray(i)) for i in eta_index.flatten()])
    IMG = np.reshape(RGBA, IMG.shape)

    kk = np.minimum(1, qw / 5.)

    IMG[:,:,0] = np.minimum(1, (0*kk + (1-kk) * IMG[:,:,0]))
    IMG[:,:,1] = np.minimum(1, (0.72*kk + (1-kk) * IMG[:,:,1]))
    IMG[:,:,2] = np.minimum(1, (1*kk + (1-kk) * IMG[:,:,2]))

    return IMG
    
    

def elevation_colormap(): 
    '''
    Produces colormap for plotting elevations
    
    Register the colormap with:
    plt.register_cmap(name='eta_cmap', cmap=eta_cmap)
    '''   
    
    cdict = {'red':   ((0.0,  0.0, 0.0),
                       (0.7,  0.0, 0.0),
                       (0.85, 0.95, 0.95),
                       (0.92,  0.35, 0.35),
                       (1.0,  0.65, 0.65)),

             'green': ((0.0,  0.0, 0.0),
                       (0.3, 0.0, 0.0),
                       (0.7, 1.0, 1.0),
                       (0.85,  0.92, 0.92),
                       (0.92,  0.60, 0.6),
                       (1.0,  0.4, 0.4)),

             'blue':  ((0.0,  0.0, 0.0),
                       (0.4,  0.9, 0.9),
                       (0.7,  1.0, 1.0),
                       (0.85,  0.79, 0.79),
                       (0.92,  0.1, 0.1),
                       (1.0,  0.0, 0.0))}
                       
    
    eta_cmap = LinearSegmentedColormap('eta_cmap', cdict)
    plt.register_cmap(name='eta_cmap', cmap=eta_cmap)
    
    print "Created new colormap 'eta_cmap'"
    