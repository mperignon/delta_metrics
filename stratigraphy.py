import numpy as np
from profiles import radial_profile
from scipy import signal

def sand_body_properties(strata_IMG):
    '''
    Calculate sand body properties

    Input:
    ------
    strata_IMG: 2D vertical slice of stratigraphy,
                possibly from radial_stratigraphy()

    Output:
    -------
    corrL: list of vertical lag and correlation
    corrW: list of horizontal lag and correlation
    '''


    A = strata_IMG > 0.5
    A = signal.medfilt2d(A.astype(float), 3)
    L, W = A.shape

    M = np.mean(A)
    S = np.std(A)


    scale = 0.025
    lag_max = int(5/scale)
    ACFL = np.zeros((lag_max + 1,))

    for lag in range(lag_max+1):
        ACFL[lag] = (np.mean((A[:L-lag,:] - M) * (A[lag:,:] - M)) / S**2 *
                     (L - lag) * W / ((L - lag) * W - 1))

    corrmatL = ACFL / 5
    lagL = np.arange(0, lag_max*scale + scale, scale)


    scale = 100.
    lag_max = int(3000/scale)
    ACFW = np.zeros((lag_max + 1,))

    for lag in range(lag_max+1):
        ACFW[lag] = (np.mean((A[:,:W-lag] - M) * (A[:,lag:] - M)) / S**2 *
                     (W - lag) * L / ((W - lag) * L - 1))

    corrmatW = ACFW / 5
    lagW = np.arange(0, lag_max*scale + scale, scale)

    return [lagL, corrmatL], [lagW, corrmatW]






def radial_stratigraphy(strata,
                        radii = [0.1, 0.3, 0.5],
                        centerpoint = None,
                        density = 0.5):
    '''
    Extracts slices of the stratigraphy at given radii from a centerpoint
    
    Input:
    -------
    strata: (m * n * 3) numpy array of stratigraphy
    radii: sequence of radii (as fraction of length of first axis of strata)
    centerpoint: tuple of x and y indices of circle certerpoint
    density: multiplier for interpolation line
    
    Output:
    --------
    profiles: list of numpy arrays of stratigraphy for each radii, in same order
    
    '''
    
    if centerpoint is None:

        xcenter = strata.shape[1]/2
        ycenter = 2
        centerpoint = (xcenter, ycenter)

        
    profiles = []

    for r in radii:

        profile_store = []

        for i in range(strata.shape[2]):

            image = strata[:,:,i]
            profile = radial_profile(image,
                                     [r],
                                     centerpoint = centerpoint,
                                     density = density)
            profile_store.append(profile[r][1])

        profiles.append(np.flipud(np.array(profile_store)))
        
    return profiles
    
    
    
    

def sedimentograph(strata, step_radius = 5):

    zmaxn = strata.shape[2]

    min_radius = 5
    max_radius = strata.shape[0]/2
    step_radius = step_radius

    # centerpoint of circle slice
    x0, y0 = 2, strata.shape[0]/2


    len_arrays = (max_radius - min_radius) / step_radius + 1
    sand_vol, mud_vol, rad_dist = np.zeros((3, len_arrays))


    for n,r0 in enumerate(range(min_radius, max_radius+1, step_radius)):

        Ncol = int(np.pi * r0) * 2
        circslice = np.zeros((Ncol,zmaxn + 1)) - 1

        for col in range(1, Ncol):

            tht = col / float(Ncol) * np.pi - np.pi/2
            ic = int(x0 + np.cos(tht) * r0)
            jc = int(y0 + np.sin(tht) * r0)

            for z in range(zmaxn):
                circslice[col-1,z] = strata[ic-1,jc-1,z]

        circslice = np.fliplr(circslice)


        sand_vol[n] = np.maximum(0,circslice).sum()
        mud_vol[n] = (1-np.abs(circslice)).sum()
        rad_dist[n] = r0


    seds = {}

    seds['norm_distance_from_apex'] = rad_dist / 100.
    seds['sand_frac_by_vol'] = sand_vol / (sand_vol + mud_vol)
    seds['sand_vol'] = sand_vol
    seds['mud_vol'] = mud_vol
    
    return seds



def sediment_volumes(strata, dx, dz, mask = None):
    
    if mask is None:
        mask = np.ones_like(strata[:,:,0])
        
    strataV = strata >= 0

    vol_total = np.sum(mask * np.sum(strataV, 2)) * dz * dx**2
    vol_sand = np.sum(mask * np.sum(np.maximum(0,strata), 2)) * dz * dx**2
    frac_sand = vol_sand / vol_total
    
    return vol_total, vol_sand, frac_sand