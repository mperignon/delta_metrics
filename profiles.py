import numpy as np
import scipy as sp
import scipy.interpolate
from skimage.graph import route_through_array


def radial_profile(image, radii, centerpoint = None, density = 2):
    '''
    Obtains profiles of image values along semi-circular arcs at distances
    given by radii away from centerpoint
    
    Input:
    --------
    image: 2D numpy array
    radii: sequence of radii for profiles, as fractions of image.shape[0])
    centerpoint: x and y indices of centerpoint. If not defined,
                 defaults to (len(x)/2, 0)
    density: (int or float) multiplier for number of points on
             interpolation line
    
    Output:
    --------
    profiles: dictionary with keys given by radii of angles and
    values of profiles
    '''

    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    
    if max(radii) > 1:
        
        radii = [r/100. for r in radii]


    if centerpoint is None:

        xcenter = len(x)/2
        ycenter = 0

    else:
        xcenter, ycenter = centerpoint



    interp = sp.interpolate.interp2d(x, y, image)
    vinterp = np.vectorize(interp)

    profiles = {}


    for rad in radii:    # radii for circles around image's center

        r = rad * image.shape[0]

        arclen = 2*np.pi*r
        angle = np.linspace(-np.pi/2, np.pi/2, int(arclen*density), endpoint=False)

        xval = xcenter + r*np.sin(angle)
        yval = ycenter + r*np.cos(angle)
        value = vinterp(xval, yval)

        profiles[rad] = [list(angle), list(value)]

    return profiles
    



def main_channel_path(image, threshold_value = 1,
                      sourcePt = None, sinkPt = None):
    '''
    Finds the path along the main channel and calculates
    the total distance along the path
    
    Input
    -------
    image: Numpy array of cost. Preferred cells should have lower values.
    threshold_value: Fraction of maximum value of image that defines which cells
                     should be always avoided.
                     Path will end if it reaches cells with 
                     value > threshold_value * image.max()
    sourcePt: (row, column) indices of start point for path
    sinkPt: (row, column) indices of target point for path.
            Path will aim for sinkPt but will stop if it reaches cells with
            values above threshold_value
            
            
    Output:
    --------
    path: Boolean numpy array of same size as image showing path
    tot_distance: Total distance (in cells) of the path
    [distances, values]: profile of values along path
    '''
    
    adj = 0

    if np.min(image) < 0:
        adj = image.min()
        image = image - adj


    threshold = image > threshold_value * image.max()

    if sourcePt is None:
        sourcePt = (0, image.shape[1]/2)
    if sinkPt is None:
        sinkPt = (image.shape[0]-2, image.shape[1]/2)

    indices, weight = route_through_array(image, sourcePt, sinkPt)
    indices = np.array(indices).T

    path = np.zeros_like(image)
    path[indices[0], indices[1]] = 1
    path[threshold] = 0

    last_index = [i for i in range(len(indices[0])) if threshold[indices[0][i], indices[1][i]]]

    if len(last_index) == 0:
        last_index = [-1]

    xy_diff = np.diff(indices[:,0:last_index[0]])**2
    xy_diff = np.sqrt(xy_diff[0] + xy_diff[1])

    tot_distance = np.sum(xy_diff)

    distances = [0] + list(np.cumsum(xy_diff))
    values = image[indices[0,:last_index[0]],indices[1,:last_index[0]]] + adj   
    
    return path, tot_distance, [distances, values]




def slice_topography(eta, heights = [0, 0.3, 0.5]):
    '''
    Creates arrays showing topography above certain heights
    
    Input:
    -------
    eta: elevation array (n * m)
    heights: sequence of elevation values (1 * k)
    
    Output:
    -------
    slices: numpy array (n * m * k) of topography slices
    '''

    if type(heights) in (float, int):
        heights = [heights]

    slices = np.zeros((eta.shape + (len(heights),)))

    for n,h in enumerate(heights):

        slices[:,:,n] = eta > h

    return slices   