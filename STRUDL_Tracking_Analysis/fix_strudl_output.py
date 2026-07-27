import numpy as np
import pandas as pd
import datetime
import glob
import pickle
import os
from helper_functions import get_stereo_heeq_lon
import json


def load_jplot(path, instrument, ftpsc, bflag, start, end):

    savepath_jplot = os.path.join(path, 'jplot', ftpsc, bflag, instrument, str(start[:4]), 'plots')
    savepath_param = os.path.join(path, 'jplot', ftpsc, bflag, instrument, str(start[:4]), 'params')

    instrument_name = 'hi1' if instrument == 'hi_1' else 'hi2'

    jplot_name = f"jplot_{instrument_name}_{start}_{end}_{ftpsc}_{bflag[0]}"
    param_fil = glob.glob(f"{savepath_param}/{jplot_name}_params.pkl")[0]

    plot_fil = glob.glob(f"{savepath_jplot}/{jplot_name}.pkl")[0]
    
    
    with open(param_fil, 'rb') as f:
        t_beg, t_end, e_beg, e_end = pickle.load(f)

    with open(plot_fil, 'rb') as f:
        img_rescale, orig = pickle.load(f)

    return img_rescale, orig, t_beg, t_end, e_beg, e_end


def get_corrected_dict(corrected_files, id_key):


    corrected_dict = {}
    for track_file in corrected_files:
        df = pd.read_csv(track_file)
        cme_id = df[id_key].iloc[0]
        times = pd.to_datetime(df['time']).tolist()
        elongs = df['elongation'].tolist()
        fcode = df['featureCode'].iloc[0]
        if cme_id not in corrected_dict:
            corrected_dict[cme_id] = {}

        corrected_dict[cme_id].update({fcode:
            {'time': times,
            'elongation': elongs,
            'featureCode': fcode}})
    
    return corrected_dict

def get_corrected_dict_combined(corrected_files):

    corrected_dict = []
    for track_file in corrected_files:
        df = pd.read_csv(track_file)

        corrected_dict.append({
            'cmeID_strudl': df['cmeID_strudl'].iloc[0],
            'cmeID_elevo': df['cmeID_elevo'].iloc[0],
            'featureCode': df['featureCode'].iloc[0],
            'time':  pd.to_datetime(df['time']).tolist(),
            'elongation': df['elongation'].tolist(),
        })
    
    return corrected_dict

def update_with_corrections(update_dict, corrected_dict, redo_df, id_key):

    if id_key == 'cmeID_strudl':
        time_215_key = 'time_21_5Rs'
        phi_key = 'phi'
        hw_key = 'halfwidth'
        vinit_key = 'vinit'

    elif id_key == 'cmeID_elevo':
        time_215_key = 'tinit [UT]'
        phi_key = 'phi [°]'
        hw_key = 'halfwidth [°]'
        vinit_key = 'vinit [km/s]'

    for j in range(len(redo_df)):
        key = redo_df[j][id_key]
        fcode = redo_df[j]['featureCode']
        if key in corrected_dict:
            if fcode in corrected_dict[key]:
                new_time = np.array([ctim.to_pydatetime() for ctim in corrected_dict[key][fcode]['time']])

                sta_lon_uncorrected = get_stereo_heeq_lon(new_time[0], sc='A')[0]
                sta_lon_corrected = get_stereo_heeq_lon(new_time[0], sc='A')[0]
                corrected_phi = sta_lon_corrected - (sta_lon_uncorrected - redo_df[j][phi_key])

                new_time = np.array([datetime.datetime.strftime(t, '%Y-%m-%dT%H:%M:%S.%f') for t in new_time])

                if id_key != 'cmeID_strudl':
                    strudl_id = None
                    pa = None
                else:
                    strudl_id = redo_df[j]['cmeID_strudl']
                    pa = redo_df[j]['pa']

                update_dict.append({
                    'cmeID_strudl': strudl_id,
                    'cmeID_elevo': redo_df[j]['cmeID_elevo'],
                    'featureCode': redo_df[j]['featureCode'],
                    'L1_ist_obs': redo_df[j]['L1_ist_obs'],
                    'phi': corrected_phi,
                    'vinit': redo_df[j][vinit_key],
                    'halfwidth': redo_df[j][hw_key],
                    'time': new_time,
                    'time_21_5Rs': redo_df[j][time_215_key],
                    'elongation': corrected_dict[key][fcode]['elongation'],
                    'pa': pa
                })
            else:
                if id_key == 'cmeID_strudl':
                    update_dict.append(redo_df[j])
                elif id_key == 'cmeID_elevo':
                    continue

        else:
            if id_key == 'cmeID_strudl':
                update_dict.append(redo_df[j])
            elif id_key == 'cmeID_elevo':
                continue
    
    return update_dict

def update_with_corrections_combined(update_dict, corrected_dict, elevo_df, strudl_df):

    for j in range(len(elevo_df)):
        key = elevo_df[j]['cmeID_elevo']
        fcode = elevo_df[j]['featureCode']

        for entry in corrected_dict:
            if entry['cmeID_elevo'] == key and entry['featureCode'] == fcode:
                new_time = np.array([ctim.to_pydatetime() for ctim in entry['time']])
                sort_order = np.argsort(new_time).astype(int)
                new_time_sorted = np.array(new_time)[sort_order]
                elon_sorted = np.array(entry['elongation'])[sort_order]

                sta_lon_uncorrected = get_stereo_heeq_lon(new_time_sorted[0], sc='A')[0]
                sta_lon_corrected = get_stereo_heeq_lon(new_time_sorted[0], sc='A')[0]
                corrected_phi = sta_lon_corrected - (sta_lon_uncorrected - elevo_df[j]['phi [°]'])


                new_time_sorted = np.array([datetime.datetime.strftime(t, '%Y-%m-%dT%H:%M:%S.%f') for t in new_time_sorted])

                pa = None
                for s in range(len(strudl_df)):
                    if key in strudl_df[s]:
                        pa = strudl_df[s]['pa']

                update_dict.append({
                    'cmeID_strudl': entry['cmeID_strudl'],
                    'cmeID_elevo': entry['cmeID_elevo'],
                    'featureCode': entry['featureCode'],
                    'L1_ist_obs': elevo_df[j]['L1_ist_obs'],
                    'phi': corrected_phi,
                    'vinit': elevo_df[j]['vinit [km/s]'],
                    'halfwidth': elevo_df[j]['halfwidth [°]'],
                    'time': new_time_sorted,
                    'time_21_5Rs': elevo_df[j]['tinit [UT]'],
                    'elongation': elon_sorted,
                    'pa': pa
                })
    
    return update_dict

def create_new_file(path_retrack_strudl, path_tracks_elevo):

    corrected_tracks_path = 'misc/corrected_tracks_new/'
    corrected_tracks_strudl = sorted(glob.glob(f"{corrected_tracks_path}/edited_track_corrected_CME_*.csv"))
    corrected_tracks_elevo = sorted(glob.glob(f"{corrected_tracks_path}/edited_track_corrected_*-CME-*.csv"))
    # read into dictionary with same format as strudl_redo

    corrected_dict_strudl = get_corrected_dict(corrected_tracks_strudl, 'cmeID_strudl')
    corrected_dict_elevo = get_corrected_dict(corrected_tracks_elevo, 'cmeID_elevo')

    strudl_redo = json.load(open(path_retrack_strudl, 'r'))
    elevo_redo = json.load(open(path_tracks_elevo, 'r'))

    updated_strudl_dict = []
    updated_strudl_dict = update_with_corrections(updated_strudl_dict, corrected_dict_strudl, strudl_redo, 'cmeID_strudl')
    updated_strudl_dict = update_with_corrections(updated_strudl_dict, corrected_dict_elevo, elevo_redo, 'cmeID_elevo')

    updated_strudl_dict_df = pd.DataFrame(updated_strudl_dict)
    updated_strudl_dict_df.to_json(path_retrack_strudl.replace('.json', '_corrected.json'), orient='records', date_format='iso')

def create_new_file_combined(path_retrack_strudl, path_tracks_elevo):

    corrected_tracks_path = 'corrected_tracks_new'
    corrected_tracks = sorted(glob.glob(f"{corrected_tracks_path}/*.csv"))

    # read into dictionary with same format as strudl_redo

    corrected_dict = get_corrected_dict_combined(corrected_tracks)

    strudl_redo = json.load(open(path_retrack_strudl, 'r'))
    elevo_redo = json.load(open(path_tracks_elevo, 'r'))

    updated_dict = []
    update_with_corrections_combined(updated_dict, corrected_dict, elevo_redo, strudl_redo)

    updated_dict_df = pd.DataFrame(updated_dict)
    updated_dict_df.to_json(path_retrack_strudl.replace('.json', '_corrected.json'), orient='records', date_format='iso')

if __name__ == "__main__":

    path_retrack_strudl = 'data/mean_45_2024_05_01_2025_05_01_strudl_tracks.json'
    path_tracks_elevo = '../ELEvo/data/elevo_results_20240501_20250501.json'

    create_new_file_combined(path_retrack_strudl, path_tracks_elevo)