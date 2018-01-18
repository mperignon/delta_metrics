import numpy as np
import cPickle as pickle
from scipy.optimize import curve_fit
from file_util import load_matlab_data

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
    difference, phi, O_phi, f_R = np.zeros((4, len(filenames,)))

    # create circular mask
    maskmap, area_mask = create_circular_mask(data, data.shape[1]/2., 0, 5, 55)

    # get base image values
    chan_base = data * maskmap
    fw_base = chan_base.sum() / area_mask
    Adry_base = area_mask * (1 - fw_base)
    


    for n,f in enumerate(filenames):

        mapfile = pickle.load( open( f, "rb" ) )
        data = mapfile['channelmap']

        chan_step = data * maskmap
        fw_step = chan_step.sum() / area_mask

        difference[n] = np.abs(chan_base - chan_step).sum()

        phi[n] = fw_base * (1 - fw_step) + (1 - fw_base) * fw_step
        O_phi[n] = 1 - difference[n] / (area_mask * phi[n])

        overlapmap = overlapmap + chan_step
        drymap = np.maximum(0, 1 - overlapmap) * maskmap
        f_R[n] = 1 - drymap.sum() / Adry_base


    return overlapmap, difference, phi, O_phi, f_R
    
    
    
def create_circular_mask(data, x0, y0, r1, r2, angle_range = (0, np.pi)):
    
    n = max(data.shape)
    y,x = np.ogrid[-y0:n-y0, -x0:n-x0]

    mask = (x*x + y*y <= r2*r2) & (x*x + y*y >= r1*r1)

    array = np.zeros((n, n))
    array[mask] = 1

    tmin,tmax = angle_range
    theta = np.arctan2(x,y) - tmin + np.pi/2
    theta %= (2*np.pi)
    anglemask = theta <= (tmax-tmin)

    anglemap = anglemask[:data.shape[0], :data.shape[1]]


    maskmap = array[:data.shape[0], :data.shape[1]] > 0

    maskmap *= anglemap
    
    area_mask = np.sum(maskmap)
    
    
    return maskmap, float(area_mask)
    
    
    
def channel_decay(froot, fbase, frange,
                  number_of_comparisons = 50,
                  average = False, mask = None):
    
    dc = number_of_comparisons
    
    
    fnums = range(frange[0], frange[1] + 1)

    fname = fbase + str(fnums[0])
    data = load_matlab_data(froot, fname)

    # create masks
    r1 = int(0.1 * data['eta'].shape[0])
    r2 = int(0.9 * data['eta'].shape[0])
    
    if mask is None:
        init_mask = np.ones_like(data['eta'])
    
    mask, area_mask = create_circular_mask(data['eta'],
                                              data['eta'].shape[1]/2., 0,
                                              r1, r2,
                                              (np.pi/6, 5*np.pi/6))

    mask = mask * init_mask


    chmap_store = []

    for n,fnum in enumerate(fnums):

        fname = fbase + str(fnum)
        data = load_matlab_data(froot, fname)

        chmap = data['uw'] > 0.3
        chmap_store.append(chmap)


    Pwet = np.zeros((dc,))
    decay = np.zeros((len(chmap_store), dc))

    for n,c in enumerate(chmap_store):

        base = float(np.sum(c * mask))
        Pwet[:] = 0

        for d in range(dc):

            try:
                step = c > chmap_store[n + d + 1]
                Pwet[d] = 1 - np.sum(step * mask)/base

            except:
                Pwet[d] = np.nan 

        decay[n,:] = Pwet
        
        
    # average decay for each time delay
    if average:
        decay = np.nanmean(decay, axis=0)
        
    return decay




def channel_decay_curve_fit(ydata, fit = 'exponential'):
    '''
    Calculates the curve fit for channel decay (averaged)
    
    Input:
    -------
    ydata: averaged remaining channel fraction (n x 1 Np.array)
           output of function channel_decay()
    fit: 'exponential', (a - b) * np.exp(-c * x) + b
         'harmonic', a / (1 + b * x)
         'linear', a * x + b
         
         
    Output:
    --------
    fit_parameters: dictionary of
        xdata: time lag
        ydata: input ydata, averaged remaining channel fraction
        fit: calculated remaning channel fraction
        parameters: fitted curve parameters
        error: one standard deviation error for parameters
        covariance: estimated covariance of parameters
    '''
    
    avail_fits = ['exponential', 'harmonic', 'linear']
    
    assert fit in avail_fits, "%s fit not available"%fit
    
    
    
    func_exponential = lambda x,a,b, c: (a - b) * np.exp(-c * x) + b
    func_harmonic = lambda x,a,b: a / (1 + b * x)
    func_linear = lambda x,a,b: a * x + b

    xdata = np.arange(1,len(ydata)+1)

    if fit == 'exponential':

        popt, pcov = curve_fit(func_exponential, xdata, ydata)
        yfit = func_exponential(xdata, *popt)

    if fit == 'harmonic':

        popt, pcov = curve_fit(func_harmonic, xdata, ydata)
        yfit = func_harmonic(xdata, *popt)
        
    if fit == 'linear':

        popt, pcov = curve_fit(func_linear, xdata, ydata)
        yfit = func_linear(xdata, *popt)


    perr = np.sqrt(np.diag(pcov))

    fit_parameters = {'xdata': xdata,
                      'ydata': ydata,
                      'fit': yfit,
                      'parameters': popt,
                      'error': perr,
                      'covariance': pcov}
    
    return fit_parameters




def channel_overlay(froot, fbase, frange):
    '''
    Creates a cumulative image of the presence of water
    
    Input:
    -------
    froot: directory containing the .mat files
    fbase: root of the filenames
    frange: sequence of start and end numbers of the filenames
    
    Example:
    ---------
    froot: '../JGR2016_data/'
    fbase: 'baseF25data'
    frange: (4001,4004)
    would open files:
    ['../JGR2016_data/baseF25data4001.mat',
    '../JGR2016_data/baseF25data4002.mat',
    '../JGR2016_data/baseF25data4003.mat',
    '../JGR2016_data/baseF25data4004.mat']
    
    Output:
    -------
    ghost_image: numpy array  
    '''

    ghost_image = None

    for fnum in range(frange[0],frange[1]+1):
        
        fname = fbase + str(fnum)
        data = load_matlab_data(froot, fname)
        
        if ghost_image is None:
            ghost_image = np.zeros(data['qw'].shape)

        ghost_image += data['qw'] / (data['qw'].max() - data['qw'].min())
        
    return ghost_image
        
