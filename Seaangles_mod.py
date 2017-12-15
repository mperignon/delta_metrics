from scipy import sparse
from scipy.spatial import ConvexHull
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np

def Seaangles_mod(numviews,thresholdimg):
    '''
    From Seaangles_mod.m
    
    Takes an image and extracts its Opening Angle Map
    
    Returns shorelines and shallowsea????
    
    '''

    
    Sx, Sy = np.gradient(thresholdimg)
    G = np.sqrt(Sx**2 + Sy**2)

    edges = (G > 0) & (thresholdimg > 0)

    bordermap = np.pad(np.zeros_like(edges), 1, 'edge')
    bordermap[:-2,1:-1] = edges
    bordermap[0,:] = 1


    points = np.fliplr(np.array(np.where(edges > 0)).T)
    hull = ConvexHull(points, qhull_options='Qc')
    
    
    
    sea = np.fliplr(np.array(np.where(thresholdimg > 0.5)).T)

    points_to_test = [Point(i[0],i[1]) for i in sea]
    polygon = Polygon(points[hull.vertices]).buffer(0.01)

    In = np.array(map(lambda pt: polygon.contains(pt), points_to_test))
    Shallowsea_ = sea[In]

    seamap = np.zeros(bordermap.shape)
    flat_indices = map(lambda x: np.ravel_multi_index(x,seamap.shape),
                                 np.fliplr(Shallowsea_))
    seamap.flat[flat_indices] = 1
    seamap[:3,:] = 0

    
    Deepsea_ = sea[~In]
    Deepsea = np.zeros((7,len(Deepsea_)))
    Deepsea[:2,:] = np.flipud(Deepsea_.T)
    Deepsea[-1,:] = 200. # where does this 200 come from?


    Shallowsea = np.array(np.where(seamap > 0.5))
    shoreandborder = np.array(np.where(bordermap > 0.5))

    c1 = len(Shallowsea[0])
    c2 = len(shoreandborder[0])
    maxtheta = np.zeros((numviews,c1))

    for i in range(c1):

        diff = shoreandborder - Shallowsea[:, i, np.newaxis]
        x = diff[0]
        y = diff[1]

        angles = np.arctan2(x,y)
        angles = np.sort(angles) * 180. / np.pi

        dangles = angles[1:] - angles[:-1]
        dangles = np.concatenate((dangles, [360 - (angles.max() - angles.min())]))
        dangles = np.sort(dangles)

        maxtheta[:,i] = dangles[-numviews:]

        
        
    allshore = np.array(np.where(edges > 0))
    c3 = len(allshore[0])
    maxthetashore = np.zeros((numviews,c3))

    for i in range(c3):

        diff = shoreandborder - allshore[:, i, np.newaxis]
        x = diff[0]
        y = diff[1]

        angles = np.arctan2(x,y)
        angles = np.sort(angles) * 180. / np.pi

        dangles = angles[1:] - angles[:-1]
        dangles = np.concatenate((dangles, [360 - (angles.max() - angles.min())]))
        dangles = np.sort(dangles)

        maxthetashore[:,i] = dangles[-numviews:]

        
        
    waves1 = np.vstack([np.hstack([Shallowsea, Deepsea[:2,:]]),
                        np.hstack([maxtheta.sum(axis=0), Deepsea[-1,:]])])
    
    waves1s = sparse.csr_matrix((waves1[2,:],(waves1[0,:], waves1[1,:])),
                                shape=thresholdimg.shape)

    
    
    shoreline = np.vstack([allshore, maxthetashore.sum(axis=0)])

    picshore = sparse.csr_matrix((shoreline[2,:],(shoreline[0,:], shoreline[1,:])),
                                shape=thresholdimg.shape)

    
    
    shoreangles = np.vstack([allshore, maxthetashore])
    seaangles = np.hstack([np.vstack([Shallowsea, maxtheta]),Deepsea])
    
    
    return shoreangles, waves1s, seaangles, picshore

