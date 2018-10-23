import numpy as np
from shapely.geometry import shape, Point, Polygon, MultiLineString, MultiPoint, MultiPolygon, LineString

import cPickle as pickle
from metrics_utils import *
import itertools



# def azimuth(point1, point2):
#     '''azimuth between 2 shapely points (interval 0 - 360), from vertical'''
#     
#     angle = np.arctan2(point2.x - point1.x, point2.y - point1.y)
#     az = np.degrees(angle) if angle > 0 else np.degrees(angle) + 360
#     
#     return az
#     
    

def extract_channel_props(network_lines, network_widths, bounds):

    network_min_widths = np.zeros((len(bounds),))
    network_avg_widths = np.zeros((len(bounds),))
    network_max_widths = np.zeros((len(bounds),))

    for n in range(len(bounds)):

        # channels added in open water to close island polygons have width 9999
        channels = [network_lines[b] for b in bounds[n]
                    if network_widths[b] != 9999]
        widths = [network_widths[b] for b in bounds[n]
                    if network_widths[b] != 9999]
        lengths = [c.length for c in channels]


        tot_length = sum(lengths)
        avg_width = (sum([widths[b] * lengths[b]
                          for b in range(len(widths))]) / tot_length)
        max_width = max(widths)
        min_width = min(widths)


        network_avg_widths[n] = avg_width
        network_max_widths[n] = max_width
        network_min_widths[n] = min_width
        
        
    return network_min_widths, network_avg_widths, network_max_widths
    


def calculate_channel_sinuosity(islands,
                                network_lines,
                                bounds,
                                step_distance = [500, 1000, 1500],
                                save = True,
                                load_saved = False,
                                file_root = ''):
                                
        
        
    if type(step_distance) is not list:
        step_distance = list(step_distance)
        
                                
    if load_saved:
    
        sinuosity_all = pickle.load( open(
        file_root + '/bound_channel_sinuosity_vals.p', "rb" ) )
        
        
    else:

        sinuosity_all = np.ones((len(islands),
                                 len(step_distance)))
    
    
        for cn, c_dist in enumerate(step_distance):
    
            for ni,i in enumerate(islands):
                sinuosity = np.ones((len(bounds[ni]),))

        
                for n,b in enumerate(bounds[ni]):
                    line = network_lines[b]

        
                    if line.length > c_dist:

                        l_dist = []
                    
                        for s in range(0, c_dist, 100):

                            coords = []

                            for d in np.arange(s, line.length+1, c_dist):  
                                coords.append(line.interpolate(d).coords[0])

                            diff = np.diff(np.array(coords), axis=0)
                            l_dist += list(np.sqrt(diff[:,0]**2 + diff[:,1]**2))


                        # reverse island perimeter    
                        rline = LineString(list(line.coords)[::-1]) 
                    
                        for s in range(0, c_dist, 100):

                            coords = []

                            for d in np.arange(s, rline.length+1, c_dist):  
                                coords.append(rline.interpolate(d).coords[0])

                            diff = np.diff(np.array(coords), axis=0)
                            l_dist += list(np.sqrt(diff[:,0]**2 + diff[:,1]**2))

                        
                        sin = np.median(c_dist / np.array(l_dist))
                        sinuosity[n] = sin


                sinuosity_all[ni,cn] = np.mean(sinuosity)
                
        if save:
        
            pickle.dump(bounds, open(
                        file_root + '/bound_channel_sinuosity_vals.p', "wb"))
            
            
    return sinuosity_all

    
    
    
def calculate_channel_junction_stats(network_lines,
                                     bounds,
                                     save = True,
                                     load_saved = False,
                                     file_root = ''):
    
    
    if load_saved:
    
        angle_stats = pickle.load(open(
                        file_root + '/bound_channel_angle_stats.p', "rb"))

    else:

        angle_stats = []

        for b in bounds:

            val = []
            freq = []

            lines = [network_lines[l] for l in b]
            lengths = [l.length/2 for l in lines]

            if len(lines)>1:

                for l1,l2 in itertools.combinations(range(len(lines)),2):

                    if lines[l1].touches(lines[l2]):

                        node = lines[l1].intersection(lines[l2])

                        if node.type is 'MultiPoint':
                            node = node[0]

                        buffer_l = np.min([lines[l1].length * 0.6,
                                           lines[l2].length * 0.6,
                                           500])

                        intersect0 = lines[l1].intersection(
                                        node.buffer(buffer_l).exterior)
                                        
                        if intersect0.type is 'MultiPoint':
                            intersect0 = intersect0[0]

                        az0 = azimuth(node, intersect0)




                        intersect1 = lines[l2].intersection(
                                        node.buffer(buffer_l).exterior)
                                        
                        if intersect1.type is 'MultiPoint':
                            intersect1 = intersect1[0]

                        az1 = azimuth(node, intersect1)

                        az = np.abs(az0 - az1)
                        diff_az = az if az<180 else az-180

                        frac_length = (lengths[l1] + lengths[l2])

                        freq.append(frac_length)
                        val.append(diff_az)

                val = np.array(val)
                freq = np.array(freq)
                
                # mean
                mean = np.average(val, weights = freq)
                
                # median
                order = np.argsort(val)
                cdf = np.cumsum(freq[order])
                median = val[order][np.searchsorted(cdf, cdf[-1] // 2)]
                
                # mode
                mode = val[np.argmax(freq)]
                
                # var
                dev = freq * (val - mean) ** 2
                var = dev.sum() / (freq.sum() - 1)
                
                #std
                std = np.sqrt(var)

                angle_stats.append([mean, median, mode, var, std])
                
            else:

                angle_stats.append([0,0,0,0,0])
                
                
        angle_stats = np.array(angle_stats)
        
        
        if save:
        
            pickle.dump(bounds, open(
                        file_root + '/bound_channel_angle_stats.p', "wb"))
            
            
    return angle_stats
    
    
    
    

def extract_outflow_channel_props(islands,
                                  network_lines,
                                  interior_channels,
                                  save = True,
                                  load_saved = False,
                                  file_root = ''):
    
    
    if load_saved:
        
        outflow_angles = pickle.load(
                         open( file_root + '/outflow_angles.p', "rb" ) )
                         
        outflow_stats = pickle.load(
                        open( file_root + '/outflow_stats.p', "rb" ) )
                        
        num_outflow = pickle.load(
                      open( file_root + '/outflow_number.p', "rb" ) )
        
        
    else:

        outflow_angles = []
        num_outflow = np.zeros((len(islands),))

        for n in range(len(islands)):

            lines = [i for i in interior_channels[n]]

    
    
            outflow = []

            for l in lines:
                if islands[n].exterior.touches(network_lines[l]):
                    outflow.append(l)


            angle = []

            num_out = 0

            for l in outflow:

                inside_line = islands[n].intersection(network_lines[l])
                node = islands[n].boundary.intersection(network_lines[l])


                if inside_line.type is 'LineString':

                    if node.type is 'LineString':
                        node = MultiPoint(node.coords)

                    if node.type is 'MultiPoint':

                        dists = np.array([nn.distance(islands[n].exterior)
                                          for nn in node])
                        loc = np.where(dists == dists.min())[0][0]
                        node = node[loc]

                    buffer_l = np.min([inside_line.simplify(100).length * 0.8,
                                       250])
            
                    node_b = node.buffer(buffer_l).exterior

                    try:
                        intersect0 = inside_line.intersection(node_b)
                
                        if intersect0.type is 'MultiPoint':
                            intersect0 = intersect0[0]    
                        az0 = azimuth(node, intersect0)

                    except:
                        intersect0 = Point(inside_line.coords[-1])

                        if intersect0.intersects(node.buffer(10)):
                            intersect0 = Point(inside_line.coords[0])

                        az0 = azimuth(node, intersect0)


                    b = islands[n].boundary
                    outside_intersects = b.intersection(node_b)

                    intersect1 = outside_intersects[0]
                    intersect2 = outside_intersects[1]

                    az3 = azimuth(intersect1, intersect2)

                    az = np.abs(az0 - az3)

                    diff_az = az if az<180 else az - 180
                    diff_az = diff_az if diff_az<90 else 180-diff_az

                    angle.append(diff_az)
                    num_out += 1

            outflow_angles.append(angle)

            num_outflow[n] = num_out


        flat_angles = np.array([item for sublist in outflow_angles
                                for item in sublist])
                                
        outflow_stats = np.array([[np.min(m),
                                   np.max(m),
                                   np.mean(m),
                                   np.median(m),
                                   np.std(m)]
                                    if len(m)>0 else [0,0,0,0,0]
                                    for m in outflow_angles])


        if save:

            pickle.dump(outflow_angles,
                        open( file_root + '/outflow_angles' + '.p', "wb" ) )
                        
            pickle.dump(outflow_stats,
                        open( file_root + 'outflow_stats' + '.p', "wb" ) )
                        
            pickle.dump(num_outflow,
                        open( file_root + 'outflow_number' + '.p', "wb" ) )
    

    
    return outflow_angles, outflow_stats, num_outflow