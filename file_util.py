import numpy as np
from scipy.io import loadmat

def load_matlab_data(froot,fname):
    
    matdata = loadmat(froot + fname + '.mat', squeeze_me=True)['data']    
    names = matdata.dtype.fields.keys()
    
    data = {}
    
    for n in names:    
        data[n] = matdata[n][()]
    
    return data