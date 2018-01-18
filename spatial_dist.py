import numpy as np

def quantity_histograms(data,
                        min_bin = -3, max_bin = 1, dbin = 0.025,
                        radius1 = 3, radius2 = 30, radius3 = 90,
                        angle = np.pi/6.):

    mask = sector_mask(data.shape, (0, data.shape[1]/2),
                       radius3, radius1, (angle, np.pi - angle))

    mask_A = sector_mask(data.shape, (0, data.shape[1]/2),
                         radius3, radius1, (angle, np.pi/2.))
    mask_As = sector_mask(data.shape, (0, data.shape[1]/2),
                          radius3, radius2, (angle, np.pi/2.))

    mask_B = sector_mask(data.shape, (0, data.shape[1]/2),
                         radius3, radius1, (np.pi/2, np.pi - angle))
    mask_Bs = sector_mask(data.shape, (0, data.shape[1]/2),
                          radius3, radius2, (np.pi/2, np.pi - angle))


    masks = [mask, mask_A, mask_As, mask_B, mask_Bs]
    
    bins = np.arange(min_bin - dbin/2, max_bin + dbin, dbin)
    hists = np.zeros((len(bins)-1, len(masks)))
    cumhist = np.zeros((len(bins)-1, len(masks)))

    for n,m in enumerate(masks):

        hist_data = data.copy()
        hist_data[~m] = -10

        count, bins = np.histogram(hist_data[hist_data > min_bin],
                                    bins, normed = True)
        hists[:,n] = count
        cumhist[:,n] = np.cumsum(count) * dbin

    nbins = bins[:-1] + dbin/2.


    return nbins, hists, cumhist, masks
    
    
    
def sector_mask(shape,centre,radius1, radius2 = 0, angle_range = (0, np.pi)):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = angle_range

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = (r2 <= radius1*radius1) & (r2 >= radius2*radius2)

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return circmask*anglemask