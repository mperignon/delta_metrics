import numpy as np

def sedimentograph(strata, step_radius = 5):

    zmaxn = strata.shape[2]

    min_radius = 5
    max_radius = strata.shape[0]/2
    step_radius = step_radius

    # centerpoint of circle slice
    x0, y0 = 3, strata.shape[0]/2


    len_arrays = (max_radius - min_radius) / step_radius + 1
    sand_vol, mud_vol, rad_dist = np.zeros((3, len_arrays))


    for n,r0 in enumerate(range(min_radius, max_radius+1, step_radius)):

        Ncol = int(np.pi * r0) * 2
        circslice = np.zeros((Ncol,zmaxn + 1)) - 1

        for col in range(1,Ncol):

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

