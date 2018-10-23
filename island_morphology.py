
import numpy as np
from glob import glob

from osgeo import ogr
import fiona


import pandas as pd

from metrics_scripts.metrics_utils import *
from metrics_scripts.island_creation_utils import *
from metrics_scripts.island_metric_utils import *
from metrics_scripts.channel_metric_utils import *


file_root = 'ganges_metrics/metrics_results'

island_filename = 'ganges_metrics/islands.tif'
network_filename = 'ganges_metrics/network.shp'

island_tiff_filename = 'ganges_metrics/gangeschan.tif'

save = False
load_saved = True
load_raw_island_shapefile = False



print 'Loading island tiff'

ganges_mask, params = read_tiff_as_array(island_tiff_filename,
                                         get_info = True,
                                         normalize = True)
                                         
r_xmax, r_xmin, r_ymax, r_ymin, r_xres, r_yres = params
pixel_size = r_xres

downsample = 1

if downsample > 1:
    ganges_mask = ganges_mask[::downsample,::downsample]
    pixel_size = r_xres * downsample




print 'Loading network shapefile'

parameters = ['Width']

network_lines, network_params = load_shapefile(network_filename, parameters)

poly_islands = polygonize_shapely_lines(network_lines,
                                   size_threshold = pixel_size **2 * 10)



print 'Finding nested islands'

poly_islands, contained_islands = find_nested_islands(poly_islands,
                                                 merge_islands = True)




print 'Finding interior channels'

bounds, interior_channels = find_bounding_channels(network_lines,
                                                   poly_islands,
                                                   save = save,
                                                   load_saved = load_saved,
                                                file_root = 'ganges_metrics')

flat_bounds = np.unique([item for sublist in bounds for item in sublist])



# channel metrics

print 'Calculating channel metrics: channel width'

net_avg_widths, net_max_widths = extract_channel_props(network_lines,
                                                       network_params['Width'],
                                                       bounds)


print 'Calculating channel metrics: channel angles'

channel_angles = calculate_channel_junction_stats(network_lines,
                                                 bounds,
                                                 save = save,
                                                 load_saved = load_saved,
                                                 file_root = file_root)

print 'Calculating channel metrics: channel sinuosity'

sinuosity = calculate_channel_sinuosity(poly_islands,
                                network_lines,
                                bounds,
                                step_distance = [500, 1000, 1500],
                                save = save,
                                load_saved = load_saved,
                                file_root = file_root)


print 'Load island shapefile'


if load_raw_island_shapefile:

    islands = load_raw_island_shapefile(island_filename,
                                        poly_islands,
                                        save_shapefile = True,
                                        file_root = 'ganges_metrics')

else: 


    islands, _ = load_shapefile(island_filename[:-4] + '_sorted.shp')




print 'Smoothing islands'

smooth_islands = get_smooth_islands(islands)





print 'Calculating channel metrics: outflow channels'

out_angles, out_stats, num_outflow = extract_outflow_channel_props(
                                                        smooth_islands,
                                                        network_lines,
                                                        interior_channels,
                                                        save = save,
                                                        load_saved = load_saved,
                                                        file_root = file_root)




print 'Calculating island metrics: polygon metrics'

polygon_metrics = extract_polygon_props(smooth_islands,
                                        network_lines,
                                        interior_channels)


print 'Calculating island metrics: edge distances'

edgedists = calculate_edge_distances(smooth_islands,
                                     save = save,
                                     load_saved = load_saved,
                                     file_root = file_root)




print 'Assigning zones'

files = glob('../../GIS/GBMD_network_data/zones/t*.shp')
zone_cat = load_zone_shapefiles(islands, files)



print 'Saving properties shapefile'

new_props = {'avg_width':net_avg_widths,
             'edge_d2': edgedists,
             'max_width':net_max_widths,
             'o_ang_min': out_stats[:,0],
             'o_ang_max': out_stats[:,1],
             'o_ang_mean': out_stats[:,2],
             'o_ang_med': out_stats[:,3],
             'o_ang_std': out_stats[:,4],
             'out_numbr': num_outflow,
             'sin500': sinuosity[:,0],
             'sin1000': sinuosity[:,1],
             'sin1500': sinuosity[:,2],
             'zone': zone_cat
             }

fields = polygon_metrics.copy()
fields.update(new_props)

field_type = {}
for k in fields.keys():
    field_type[k] = ogr.OFTReal
    
    
create_shapefile_from_shapely_multi(islands,
                        'ganges_metrics/metrics_results/islands_properties.shp',
                        fields = fields,
                        field_type = field_type)


# 
# fields['sin500'][np.where(fields['sin500']>1.115)[0]] = 1.12
# fields['sin1000'][np.where(fields['sin1000']>1.1)[0]] = 1.1
# fields['sin1500'][np.where(fields['sin1500']>1.12)[0]] = 1.12
# 
# 
# classify_data = pd.DataFrame()
# 
# classify_data['area'] = np.log10(fields['p_area'] /
#                                  fields['p_area'].min())
#                                  
# classify_data['avg_width'] = np.log10(fields['avg_width'] /
#                                       fields['avg_width'].min())
#                                       
# classify_data['max_width'] = np.log10(fields['max_width'] /
#                                       fields['max_width'].min())
#                                       
# 
# classify_data['w_shape'] = np.log10(fields['p_w_shapef'] /
#                                     fields['p_w_shapef'].min())
#                                     
# classify_data['d_shape'] = np.log10(fields['p_d_shapef'] /
#                                     fields['p_d_shapef'].min())
#                                     
# classify_data['asp_rat'] = np.log10(fields['p_asp_rat'] /
#                                     fields['p_asp_rat'].min())
# 
# 
# classify_data['out_numbr'] = np.log10(fields['out_numbr'] + 0.1)
# 
# 
# classify_data['s500'] = np.log10(fields['sin500'] / fields['sin500'].min())
# classify_data['s1000'] = np.log10(fields['sin1000'] / fields['sin1000'].min())
# classify_data['s1500'] = np.log10(fields['sin1500'] / fields['sin1500'].min())
# 
# classify_data['convex'] = np.log10(fields['p_convex'] /
#                                    fields['p_convex'].min())
#     
#     
#     
# min_max_scaler = MinMaxScaler()
# scaled_data = min_max_scaler.fit_transform(classify_data.values)
# 
# for n,c in enumerate(classify_data.columns):
#     classify_data[c] = scaled_data[:,n]

