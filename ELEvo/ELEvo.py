
from datetime import datetime
import matplotlib.dates as mdates
import numpy as np
import time
import os
import warnings
import pandas as pd
import data_utils
import pred_utils
import copy
import logging

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')
ffmpeg_path=''

def fun_wrapper(dict_args):
    return pred_utils.Prediction_ELEvo(**dict_args)

def save_test_output(dat, res):

    nam = dat['associatedCMEID'].replace(':','_').replace('-','_')
    save_test_dict = copy.deepcopy(dat)

    (time2_num, cme_r, cme_lat,
     cme_lon, cme_a, cme_b,
     cme_c, cme_id, cme_v,
     halfwidth, arr_time_fin, arr_time_err0,
     arr_time_err1, arr_id, arr_hit,
     arr_speed_list, arr_speed_err_list) = res

    save_results_dict = {}
    save_results_dict["hc_time_num1"]= time2_num
    save_results_dict["hc_r1" ]= cme_r
    save_results_dict["hc_lat1" ]= cme_lat
    save_results_dict["hc_lon1" ]= cme_lon
    save_results_dict["a1_ell" ]= cme_a
    save_results_dict["b1_ell" ]= cme_b
    save_results_dict["c1_ell" ]= cme_c
    save_results_dict["hc_id1" ]= cme_id
    save_results_dict["hc_v1" ]= cme_v
    save_results_dict["halfwidth" ]= halfwidth
    save_results_dict["hc_arr_time1"] = arr_time_fin
    save_results_dict["hc_err_arr_time_min1"] = arr_time_err0
    save_results_dict["hc_err_arr_time_max1"] = arr_time_err1
    save_results_dict["hc_arr_id1"] = arr_id
    save_results_dict["hc_arr_hit1"] = arr_hit
    save_results_dict["hc_arr_speed1"] = arr_speed_list
    save_results_dict["hc_err_arr_speed1"] = arr_speed_err_list

    np.save('data/test_elevo_input_'+nam+'.npy',save_test_dict)

    np.save('data/test_elevo_output_'+nam+'.npy',save_results_dict)
    
def main(path_to_donki, path_to_positions, dates, strudl_path=None):

    object_list = list(path_to_positions.keys())
    data = data_utils.load_donki(path_to_donki,dates)        

    positions = {}

    for sc_obj in object_list:
        _, sc_name = data_utils.parse_space_obj_names(sc_obj)

        if not os.path.isfile(path_to_positions[sc_obj]):
            print(f'Position file for {sc_name} not found at {path_to_positions[sc_obj]}')
            print(f'Attempting to download positions for {sc_name} from JPL Horizons...')
            data_utils.create_positions_file(sc_obj, start='2010-01-01', stop='2030-12-31', step='10m', save_path=path_to_positions[sc_obj])
            #path_to_positions[sc_obj] = os.path.join(path_to_positions[sc_obj], f'positions_{sc_name}_from_2010_to_2030_HEEQ_10m_rad_mb.p')

        positions[sc_name] = data_utils.load_positions_jpl(path_to_positions[sc_obj],[mdates.date2num(dates[0]),mdates.date2num(dates[1])],sc_name)
    
    if strudl_path is not None:
        # TODO: make this work with multiple CMEs
        strudl_tracks = np.load(strudl_path, allow_pickle=True).item()
        interesting_cme = 'CME_60'
        strudl_track = strudl_tracks[interesting_cme]
        print(strudl_track)

    #print('Generating kinematics using ELEvo')

    start_time = time.time()
    
    results = []

    for d in data:

        d["positions"]=positions
        d["seed_value"]=42  # for reproducibility in tests

        res = fun_wrapper(d)
        if res != {}:
            results.append(res)

    results_df = pd.DataFrame(results)
    results_df.to_json('data/elevo_results_'+datetime.strftime(dates[0], '%Y%m%d')+'_'+datetime.strftime(dates[1], '%Y%m%d')+'.json', orient='records', date_format='iso')

    print('Done in: ',np.round((time.time()-start_time)), 'seconds')

if __name__ == '__main__':

    dates = [datetime(2024,5,1),datetime(2025,5,1)]
    path_to_donki = '../STRUDL_Tracking_Analysis/data/DONKI_cme_analyses_matched.json'
    strudl_path = None#'data/tracks_with_parameters_mean_45_2024_05_01_2025_04_30_earth_pa_6h_cleaned.npy'


    path_to_positions = {'l1': 'data/positions_l1_from_2010_to_2030_HEEQ_10m_rad_mb.p',
                         'solo': 'data/positions_solo_from_2010_to_2030_HEEQ_10m_rad_mb.p',
                         'psp': 'data/positions_psp_from_2010_to_2030_HEEQ_10m_rad_mb.p',
                         'sta': 'data/positions_sta_from_2010_to_2030_HEEQ_10m_rad_mb.p',
                         'bepi': 'data/positions_bepi_from_2010_to_2030_HEEQ_10m_rad_mb.p',
                         'mercury': 'data/positions_mercury_from_2010_to_2030_HEEQ_10m_rad_mb.p',
                         'venus': 'data/positions_venus_from_2010_to_2030_HEEQ_10m_rad_mb.p',
                         'mars': 'data/positions_mars_from_2010_to_2030_HEEQ_10m_rad_mb.p'}

    
    main(path_to_donki, path_to_positions, dates, strudl_path)


