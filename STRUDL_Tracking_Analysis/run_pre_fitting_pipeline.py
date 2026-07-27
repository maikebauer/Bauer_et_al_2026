import helper_functions as helper_functions
import os
import datetime
from sunpy.coordinates.ephemeris import get_horizons_coord
import pandas as pd    
import numpy as np


def get_combined_cme_ips_df(file_path_donki_cmes, file_path_donki_ips):
    
    """
    """

    if not os.path.exists(file_path_donki_cmes):
        os.makedirs(os.path.dirname(file_path_donki_cmes), exist_ok=True)
        helper_functions.download_donki_data(file_path_donki_cmes, 'CME')

    if not os.path.exists(file_path_donki_ips):
        os.makedirs(os.path.dirname(file_path_donki_ips), exist_ok=True)
        helper_functions.download_donki_data(file_path_donki_ips, 'IPS')

    donki_df_cmes = pd.read_json(file_path_donki_cmes)
    donki_df_ips = pd.read_json(file_path_donki_ips)

    joint_df = helper_functions.join_donki_cme_ips_data(donki_df_cmes, donki_df_ips)

    analyses_only_df = helper_functions.get_cme_analyses_df(joint_df)

    return analyses_only_df

def get_donki_strudl_match_json(file_path_strudl_segmentation, analyses_only_df, sc='A', threshold_minutes=60*6):

    """
    """

    # Get start times of STRUDL CMEs to associate with DONKI later
    df_strudl = helper_functions.convert_strudl_output_start_time_only(file_path_strudl_segmentation, sc=sc)
    

    final_df = helper_functions.get_matching_cmes(df_pred=df_strudl, df_analyses=analyses_only_df, threshold=threshold_minutes)

    return final_df

def make_elevohi_input_tracks(file_path_strudl_segmentation, output_file_strudl_tracks_with_parameters, file_path_donki_joint, sc='A'):
    
    """
    """


    os.makedirs(os.path.dirname(output_file_strudl_tracks_with_parameters), exist_ok=True)
    strudl_track_dictionary = helper_functions.convert_strudl_output(file_path_strudl_segmentation, pa_array=None, use_earth_pa=True, use_max_elon_pa=False, sc=sc)

    donki_df = pd.read_json(file_path_donki_joint)

    stereo_heeq_lons = helper_functions.get_stereo_heeq_lon([strudl_track_dictionary[key]['time'][0] for key in strudl_track_dictionary.keys()], sc='A')
    stereo_heeq_lons_dict = {key: stereo_heeq_lons[i] for i, key in enumerate(strudl_track_dictionary.keys())}

    strudl_tracks_with_parameters = []
    for sid in strudl_track_dictionary.keys():
        if sid in donki_df['associatedStrudlID'].values:
            associated_rows = donki_df[donki_df['associatedStrudlID'] == sid]
            for row_ind, row in associated_rows.iterrows():

                if stereo_heeq_lons_dict[sid] >=0:
                    donki_phi = stereo_heeq_lons_dict[sid] - row['longitude']
                else:
                    donki_phi = row['longitude'] - stereo_heeq_lons_dict[sid]

                sort_order = np.argsort(strudl_track_dictionary[sid]['time']).astype(int)
                time_sorted = np.array(strudl_track_dictionary[sid]['time'])[sort_order]
                elon_sorted = np.array(strudl_track_dictionary[sid]['elongation'])[sort_order]

                if len(np.unique(time_sorted)) != len(time_sorted):
                    sorted_dates_unique = []
                    sorted_elon_unique =  []

                    unique_dates, unique_counts = np.unique(time_sorted, return_counts=True)

                    for i in range(len(unique_dates)):
                        sorted_dates_unique.append(unique_dates[i])

                        if unique_counts[i] > 1:
                            sorted_elon_unique.append(np.mean(elon_sorted[np.where(time_sorted == unique_dates[i])]))
                        else:
                            sorted_elon_unique.append(elon_sorted[i])

                    sorted_dates_unique = np.array(sorted_dates_unique)
                    sorted_elon_unique = np.array(sorted_elon_unique)
                
                else:
                    sorted_dates_unique = np.copy(time_sorted)
                    sorted_elon_unique = np.copy(elon_sorted)

                strudl_tracks_with_parameters.append({'cmeID_strudl': sid,
                                                    'cmeID_elevo': row['associatedActivityID'],
                                                    'featureCode': row['featureCode'],
                                                    'L1_ist_obs': row['earthArrivalTime'],
                                                    'phi': donki_phi,
                                                    'vinit': row['speed'],
                                                    'halfwidth': row['halfAngle'],
                                                    'time_21_5Rs': row['time21_5'],
                                                    'time':sorted_dates_unique,
                                                    'elongation':sorted_elon_unique,
                                                    'pa':strudl_track_dictionary[sid]['pa']
                                                    })

    strudl_tracks_with_parameters = pd.DataFrame(strudl_tracks_with_parameters)
    strudl_tracks_with_parameters.to_json(output_file_strudl_tracks_with_parameters, orient='records', date_format='iso')


def get_corrected_initial_distance(donki_df, timeframe=[]):

    donki_df['rinitCorrected'] = pd.Series([None]*len(donki_df))
    donki_df['halfAngleCorrected'] = pd.Series([None]*len(donki_df))

    rinit_corrected_list = []
    hw_corrected_list = []
    lon_corrected_list = []

    if len(timeframe) == 2:
        start_time = datetime.datetime.strptime(timeframe[0], '%Y-%m-%dT%H:%MZ')
        end_time = datetime.datetime.strptime(timeframe[1], '%Y-%m-%dT%H:%MZ')
        considered_times = donki_df[(donki_df['time21_5'] >= timeframe[0]) & (donki_df['time21_5'] <= timeframe[1]) & (donki_df['hasEarthArrival'] | donki_df['arrivalEllipsePossible']) & (donki_df['longitude'].notna()) & (donki_df['latitude'].notna()) & (donki_df['halfAngle'].notna())]['time21_5'].values

    else:
        start_time = None
        end_time = None
        considered_times = donki_df[(donki_df['hasEarthArrival'] | donki_df['arrivalEllipsePossible']) & (donki_df['longitude'].notna()) & (donki_df['latitude'].notna()) & (donki_df['halfAngle'].notna())]['time21_5'].values

    considered_times = [datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%MZ') for time_str in considered_times]
    earth_coords = get_horizons_coord('399', considered_times)

    stereo_coords = get_horizons_coord('STEREO-A', considered_times)

    valid_row_count = 0
    for row_id, row in donki_df.iterrows():

        edge_distance = 21.5                  
        if row['time21_5'] == 'NaT' or pd.isna(row['longitude']) or pd.isna(row['latitude']) or pd.isna(row['halfAngle']):
            rinit_corrected_list.append(None)
            hw_corrected_list.append(None)
            lon_corrected_list.append(None)
            continue

        if start_time and end_time:
            time_date = datetime.datetime.strptime(row['time21_5'], '%Y-%m-%dT%H:%MZ')
            if time_date < start_time or time_date > end_time:
                rinit_corrected_list.append(None)
                hw_corrected_list.append(None)
                lon_corrected_list.append(None)
                continue
        
        if not row['hasEarthArrival'] and not row['arrivalEllipsePossible']:
            rinit_corrected_list.append(None)
            hw_corrected_list.append(None)
            lon_corrected_list.append(None)

            continue
        
        print(row['associatedActivityID'])

        r_corr, hw_corr, lon_corr = helper_functions.calculate_corrected_initial_distance(edge_distance, row['halfAngle']*2, row['longitude'], row['latitude'], row['time21_5'], earth_coords[valid_row_count], make_plot=True,stereo_coords=stereo_coords[valid_row_count])
        rinit_corrected_list.append(r_corr)
        hw_corrected_list.append(hw_corr)
        lon_corrected_list.append(lon_corr)
        valid_row_count += 1

    donki_df['rinitCorrected'] = rinit_corrected_list
    donki_df['halfAngleCorrected'] = hw_corrected_list
    donki_df['longitudeCorrected'] = lon_corrected_list

    arrival_list_corrected = []
    for row_num, row in donki_df.iterrows():
        if (row['arrivalEllipsePossible'] and (row['rinitCorrected'] is None or row['rinitCorrected'] == 0)) or not row['arrivalEllipsePossible']:
            arrival_list_corrected.append(False)
        else:
            if (row['longitudeCorrected']) is not None and row['halfAngleCorrected'] is not None:
                if np.abs(row['longitudeCorrected']) < row['halfAngleCorrected']:
                    arrival_list_corrected.append(True)
                else:
                    arrival_list_corrected.append(False)

    donki_df['arrivalEllipsePossibleCorrected'] = arrival_list_corrected

    return donki_df

if __name__ == "__main__":

    file_path_strudl_segmentation = '../STRUDL/segmentation_dictionary_mean_45_2024_05_01_2025_05_01.npy'

    input_file_donki_cmes = './data/DONKI_cmes.json'
    input_file_donki_ips = './data/DONKI_ips.json'

    output_file_donki_joint = './data/DONKI_cme_analyses_matched.json'

    output_file_strudl_tracks_with_parameters = './data/mean_45_2024_05_01_2025_05_01_strudl_tracks.json'

    joint_df = get_combined_cme_ips_df(file_path_donki_cmes=input_file_donki_cmes, file_path_donki_ips=input_file_donki_ips)
    corrected_df = get_corrected_initial_distance(joint_df, timeframe=['2024-05-01T00:00Z', '2025-04-30T23:59Z'])

    matched_df = get_donki_strudl_match_json(file_path_strudl_segmentation=file_path_strudl_segmentation,
                                             analyses_only_df=corrected_df)

    if not os.path.exists(os.path.dirname(output_file_donki_joint)):
        os.makedirs(os.path.dirname(output_file_donki_joint), exist_ok=True)

    matched_df.to_json(output_file_donki_joint, orient='records')

    make_elevohi_input_tracks(file_path_strudl_segmentation=file_path_strudl_segmentation,
                              output_file_strudl_tracks_with_parameters=output_file_strudl_tracks_with_parameters,
                              file_path_donki_joint=output_file_donki_joint,
                              sc='A')

