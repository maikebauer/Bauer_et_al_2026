import helper_functions as helper_functions
import os
import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from sunpy.coordinates.ephemeris import get_horizons_coord
from sunpy.coordinates import HeliographicStonyhurst
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
import pandas as pd
import numpy as np


def get_horizons_coord_chunked(body, time_grid, max_points=9000):
    """
    get_horizons_coord for a time grid of any length.

    sunpy refuses an explicit list of 10,000 or more times and points at its start/stop/step
    dictionary form instead, but that form sets the grid bounds in TDB, which would leave the
    returned times ~69 s off the UTC grid the FOV scan indexes into. Requesting the explicit
    list in chunks keeps every returned obstime exactly on the requested grid.
    """

    if len(time_grid) < max_points:
        return get_horizons_coord(body, time_grid)

    chunks = [get_horizons_coord(body, time_grid[i:i + max_points]).spherical
              for i in range(0, len(time_grid), max_points)]

    # rebuilt rather than concatenated: astropy's concatenate rejects the chunks because each
    # one's frame carries its own obstime array, and it is deprecated in any case
    return SkyCoord(np.concatenate([chunk.lon.to_value(u.deg) for chunk in chunks]) * u.deg,
                    np.concatenate([chunk.lat.to_value(u.deg) for chunk in chunks]) * u.deg,
                    np.concatenate([chunk.distance.to_value(u.AU) for chunk in chunks]) * u.AU,
                    frame=HeliographicStonyhurst(obstime=time_grid))

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

def get_donki_strudl_match_json(file_path_strudl_segmentation, analyses_only_df, sc='A',
                                window_pad_minutes=30, threshold_fpf_minutes=60*3,
                                strudl_dict=None, central_pa_tracks=None, sta_radii_au=None):

    """
    Matches STRUDL tracks to DONKI CMEs, under both the HI-1A-entry and the Fixed-Phi criteria.

    `central_pa_tracks` and `sta_radii_au` feed the Fixed-Phi criterion: the STRUDL tracks cut
    at their central position angle, and STEREO-A's heliocentric distance per track. Leave them
    out to run the HI-1A-entry criterion alone.
    """

    # Get start times of STRUDL CMEs to associate with DONKI later
    df_strudl = helper_functions.convert_strudl_output_start_time_only(file_path_strudl_segmentation,
                                                                      sc=sc, strudl_dict=strudl_dict)

    if central_pa_tracks is not None and sta_radii_au is not None:
        df_strudl = helper_functions.add_fpf_fits(df_strudl, central_pa_tracks, sta_radii_au)

    final_df = helper_functions.get_matching_cmes(df_pred=df_strudl, df_analyses=analyses_only_df,
                                                  window_pad_minutes=window_pad_minutes,
                                                  threshold_fpf=threshold_fpf_minutes)

    return final_df

def make_elevohi_input_tracks(file_path_strudl_segmentation, output_file_strudl_tracks_with_parameters, file_path_donki_joint, sc='A', strudl_track_dictionary=None):
    
    """
    `strudl_track_dictionary` is the STRUDL tracks already cut at the Earth position angle; pass
    it in to avoid re-reading the ~1 GB segmentation file and re-querying Earth's position angle
    for every CME.
    """


    os.makedirs(os.path.dirname(output_file_strudl_tracks_with_parameters), exist_ok=True)
    if strudl_track_dictionary is None:
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


def _correct_one(task):
    """Worker for get_corrected_initial_distance; must stay module-level to be picklable."""
    edge_distance, angular_width, lon, lat, time_str, earth_coord, stereo_coord = task
    return helper_functions.calculate_corrected_initial_distance(
        edge_distance, angular_width, lon, lat, time_str, earth_coord,
        make_plot=False, stereo_coords=stereo_coord)


def get_corrected_initial_distance(donki_df, timeframe=[], n_workers=None):

    donki_df['rinitCorrected'] = pd.Series([None]*len(donki_df))
    donki_df['halfAngleCorrected'] = pd.Series([None]*len(donki_df))

    rinit_corrected_list = []
    hw_corrected_list = []
    lon_corrected_list = []

    if len(timeframe) == 2:
        start_time = datetime.datetime.strptime(timeframe[0], '%Y-%m-%dT%H:%MZ')
        end_time = datetime.datetime.strptime(timeframe[1], '%Y-%m-%dT%H:%MZ')
        considered_times = donki_df[(donki_df['time21_5'] >= timeframe[0]) & (donki_df['time21_5'] <= timeframe[1]) & (donki_df['longitude'].notna()) & (donki_df['latitude'].notna()) & (donki_df['halfAngle'].notna())]['time21_5'].values

    else:
        start_time = None
        end_time = None
        considered_times = donki_df[(donki_df['longitude'].notna()) & (donki_df['latitude'].notna()) & (donki_df['halfAngle'].notna())]['time21_5'].values

    considered_times = [datetime.datetime.strptime(time_str, '%Y-%m-%dT%H:%MZ') for time_str in considered_times]
    earth_coords = get_horizons_coord('399', considered_times)

    stereo_coords = get_horizons_coord('STEREO-A', considered_times)

    # first pass: decide which rows get corrected, in row order. The skip conditions here
    # must mirror the considered_times filter above, since valid_row_count indexes into the
    # ephemeris arrays built from it.
    tasks = []
    is_valid_row = []
    valid_row_count = 0
    for row_id, row in donki_df.iterrows():

        edge_distance = 21.5
        if row['time21_5'] == 'NaT' or pd.isna(row['longitude']) or pd.isna(row['latitude']) or pd.isna(row['halfAngle']):
            is_valid_row.append(False)
            continue

        if start_time and end_time:
            time_date = datetime.datetime.strptime(row['time21_5'], '%Y-%m-%dT%H:%MZ')
            if time_date < start_time or time_date > end_time:
                is_valid_row.append(False)
                continue

        is_valid_row.append(True)
        tasks.append((edge_distance, row['halfAngle']*2, row['longitude'], row['latitude'],
                      row['time21_5'], earth_coords[valid_row_count], stereo_coords[valid_row_count]))
        valid_row_count += 1

    # second pass: the per-CME geometry is independent, so run it across processes. 'spawn'
    # keeps each worker's VTK state clean instead of inheriting the parent's via fork.
    print(f"Correcting initial distance for {len(tasks)} CMEs on {n_workers or os.cpu_count()} workers...")
    results = []
    if tasks:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 mp_context=multiprocessing.get_context('spawn')) as executor:
            for i, result in enumerate(executor.map(_correct_one, tasks, chunksize=4), start=1):
                results.append(result)
                if i % 100 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} done")

    results_iter = iter(results)
    for valid in is_valid_row:
        r_corr, hw_corr, lon_corr = next(results_iter) if valid else (None, None, None)
        rinit_corrected_list.append(r_corr)
        hw_corrected_list.append(hw_corr)
        lon_corrected_list.append(lon_corr)

    donki_df['rinitCorrected'] = rinit_corrected_list
    donki_df['halfAngleCorrected'] = hw_corrected_list
    donki_df['longitudeCorrected'] = lon_corrected_list

    arrival_list_corrected = []
    for row_num, row in donki_df.iterrows():
        if (row['rinitCorrected'] is None or row['rinitCorrected'] == 0 or
                row['longitudeCorrected'] is None or row['halfAngleCorrected'] is None):
            arrival_list_corrected.append(False)
        else:
            arrival_list_corrected.append(bool(np.abs(row['longitudeCorrected']) < row['halfAngleCorrected']))

    donki_df['arrivalEllipsePossibleCorrected'] = arrival_list_corrected

    return donki_df


def _fov_one(task):
    """Worker for get_hi1a_fov_possible; must stay module-level to be picklable."""
    cme, time_grid, sta_coords_grid, earth_coords_grid, max_age_days, window_max_age_days = task

    _, extrapolate_rsun = helper_functions.cme_trajectory_extrapolate(cme)

    entry_time, entry_distance, early_time, late_time = helper_functions.hi1a_fov_entry_window(
        cme, time_grid, sta_coords_grid, earth_coords_grid, max_age_days=max_age_days,
        window_max_age_days=window_max_age_days)

    if entry_time is None:
        return None, None, None, None, float(extrapolate_rsun)

    return (entry_time.strftime('%Y-%m-%dT%H:%MZ'), float(entry_distance),
            early_time.strftime('%Y-%m-%dT%H:%MZ'),
            None if late_time is None else late_time.strftime('%Y-%m-%dT%H:%MZ'),
            float(extrapolate_rsun))


def get_hi1a_fov_possible(donki_df, timeframe=[], max_age_days=3.0, window_max_age_days=5.0,
                          dt_minutes=20, n_workers=None):
    """
    Marks each row of `donki_df` with whether its 3D ElEvo ellipsoid could have reached
    STEREO-A's HI-1 field of view at any point while propagating from its observed start out to
    `max_age_days` after time21_5, writing the result to a new 'couldBeInHI1AFov' column, and
    with when it first gets there, as 'predictedFirstHI1Atime' and the apex distance it has
    reached by then as 'predictedFirstHI1AdistanceRsun'.

    Since DONKI gives a single speed and that speed is uncertain, the entry time is reported as
    a window as well: 'predictedFirstHI1AtimeEarly' and 'predictedFirstHI1AtimeLate' are the
    entry times of the fastest and slowest plausible speed, which is what the STRUDL matching
    uses in place of a flat cutoff. 'predictedFirstHI1ApivotRsun' records the radius the speed
    variants were pivoted at. 'couldBeInHI1AFov' means any plausible speed gets there, i.e. the
    fastest one does, within `max_age_days`; the slower two are followed to
    `window_max_age_days` so the window stays defined for a CME that only just makes it.
    """

    valid_mask = (donki_df['longitude'].notna() & donki_df['latitude'].notna() &
                  donki_df['halfAngle'].notna() & donki_df['speed'].notna() &
                  (donki_df['time21_5'] != 'NaT'))

    if len(timeframe) == 2:
        time_in_range = ((donki_df["time21_5"] >= timeframe[0]) & (donki_df["time21_5"] <= timeframe[1]))
        valid_mask = valid_mask & time_in_range

    if not valid_mask.any():
        donki_df['couldBeInHI1AFov'] = False
        donki_df['predictedFirstHI1Atime'] = None
        donki_df['predictedFirstHI1AdistanceRsun'] = None
        donki_df['predictedFirstHI1AtimeEarly'] = None
        donki_df['predictedFirstHI1AtimeLate'] = None
        donki_df['predictedFirstHI1ApivotRsun'] = None
        return donki_df

    times21_5 = pd.to_datetime(donki_df.loc[valid_mask, 'time21_5'], format='%Y-%m-%dT%H:%MZ', utc=True)
    start_times = pd.to_datetime(donki_df.loc[valid_mask, 'associatedCMEstartTime'],format='%Y-%m-%dT%H:%MZ', utc=True, errors='coerce')

    # if there is no start time, the time21_5 is used as the start time
    scan_starts = start_times.fillna(times21_5).combine(times21_5, min)

    # builds time grid from the earliest scan start to the latest time21_5 + window_max_age_days
    grid_start = scan_starts.min()
    grid_end = times21_5.max() + pd.Timedelta(days=window_max_age_days)
    grid_index = pd.date_range(grid_start, grid_end, freq=f'{dt_minutes}min')
    time_grid = Time(grid_index.to_pydatetime())

    print(f"Fetching STEREO-A and Earth ephemerides for {len(time_grid)} time steps...")
    sta_coords_grid = get_horizons_coord_chunked('STEREO-A', time_grid)
    earth_coords_grid = get_horizons_coord_chunked('399', time_grid)

    tasks = []
    is_valid_row = []
    for row_id, row in donki_df.iterrows():
        if not valid_mask.loc[row_id]:
            is_valid_row.append(False)
            continue

        is_valid_row.append(True)

        cme = {'time21_5': row['time21_5'], 'longitude': row['longitude'],
               'latitude': row['latitude'], 'halfAngle': row['halfAngle'],
               'speed': row['speed'],
               'associatedCMEstartTime': row.get('associatedCMEstartTime')}

        # get indices of scan_start and time21_5 + window_max_age_days in the time grid
        window_start = grid_index.searchsorted(scan_starts.loc[row_id], side='left')
        window_end = grid_index.searchsorted(times21_5.loc[row_id]
                                             + pd.Timedelta(days=window_max_age_days),
                                             side='right')
        window = slice(window_start, window_end)

        # save in task, use indices to get sta1 and earth cordinates
        tasks.append((cme, time_grid[window], sta_coords_grid[window], earth_coords_grid[window],
                      max_age_days, window_max_age_days))

    print(f"Testing {len(tasks)} CMEs against the HI-1A FOV on {n_workers or os.cpu_count()} workers...")
    results = []
    if tasks:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 mp_context=multiprocessing.get_context('spawn')) as executor:
            for i, result in enumerate(executor.map(_fov_one, tasks, chunksize=4), start=1):
                results.append(result)
                if i % 200 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} done")

    results_iter = iter(results)
    entry_times = []
    entry_distances = []
    early_times = []
    late_times = []
    pivot_radii = []

    for valid in is_valid_row:
        entry_time, entry_distance, early_time, late_time, extrapolate_rsun = (
            next(results_iter) if valid else (None, None, None, None, None))
        entry_times.append(entry_time)
        entry_distances.append(entry_distance)
        early_times.append(early_time)
        late_times.append(late_time)
        pivot_radii.append(extrapolate_rsun)

    donki_df['predictedFirstHI1Atime'] = entry_times
    donki_df['predictedFirstHI1AdistanceRsun'] = entry_distances
    donki_df['predictedFirstHI1AtimeEarly'] = early_times
    donki_df['predictedFirstHI1AtimeLate'] = late_times
    donki_df['predictedFirstHI1ApivotRsun'] = pivot_radii

    # the detection criterion is that some plausible speed gets the CME into the FOV, which is
    # the fastest one, so the early edge of the window is what decides it
    donki_df['couldBeInHI1AFov'] = [early_time is not None for early_time in early_times]

    return donki_df

def save_candidate_cmes(donki_df, output_path, criterion_columns=('couldBeInHI1AFov',)):
    """
    Saves every row of `donki_df` for which any of `criterion_columns` is True to
    `output_path` as JSON, keeping all existing columns/parameters.
    """

    # check if there are any that have arrivalellipse true but are not visible in hi1a, and print a warning if so
    if 'arrivalEllipsePossibleCorrected' in donki_df.columns:
        if donki_df[(donki_df['arrivalEllipsePossibleCorrected'] == True) & (donki_df['couldBeInHI1AFov'] == False)].shape[0] > 0:
            print("Warning: There are CMEs that could arrive at L1 but are not visible in HI-1A. These will be excluded from the candidate list.")

    mask = donki_df[list(criterion_columns)].fillna(False).any(axis=1)
    candidates_df = donki_df[mask]
    candidates_df.to_json(output_path, orient='records', date_format='iso')
    return candidates_df

if __name__ == "__main__":

    file_path_strudl_segmentation = '../STRUDL/segmentation_dictionary_mean_45_2024_05_01_2025_05_01.npy'

    input_file_donki_cmes = './data/DONKI_cmes.json'
    input_file_donki_ips = './data/DONKI_ips.json'

    output_file_donki_joint = './data/DONKI_cme_analyses_matched.json'

    output_file_strudl_tracks_with_parameters = './data/mean_45_2024_05_01_2025_05_01_strudl_tracks.json'
    # output_file_candidate_cmes = './data/DONKI_cmes_candidates.json'

    timeframe = ['2024-05-01T00:00Z', '2025-04-30T23:59Z']

    joint_df = get_combined_cme_ips_df(file_path_donki_cmes=input_file_donki_cmes, file_path_donki_ips=input_file_donki_ips)

    visible_df = get_hi1a_fov_possible(joint_df, timeframe=timeframe)

    # forecasting criteria: the ecliptic correction, and the L1 arrival test derived from it
    corrected_df = get_corrected_initial_distance(visible_df, timeframe=timeframe)

    print("Loading STRUDL segmentation dictionary...")
    strudl_dict = np.load(file_path_strudl_segmentation, allow_pickle=True).item()

    central_pa_tracks = helper_functions.convert_strudl_output(use_earth_pa=False, use_central_pa=True,
                                                               sc='A', strudl_dict=strudl_dict)
    earth_pa_tracks = helper_functions.convert_strudl_output(use_earth_pa=True, sc='A',
                                                             strudl_dict=strudl_dict)


    track_keys = list(central_pa_tracks.keys())
    sta_lons, sta_radii = helper_functions.get_stereo_heeq_lon(
        [central_pa_tracks[key]['time'][0] for key in track_keys], sc='A', return_radius=True)
    sta_radii_au = dict(zip(track_keys, sta_radii))

    matched_df = get_donki_strudl_match_json(file_path_strudl_segmentation=file_path_strudl_segmentation,
                                             analyses_only_df=corrected_df,
                                             strudl_dict=strudl_dict,
                                             central_pa_tracks=central_pa_tracks,
                                             sta_radii_au=sta_radii_au)

    if not os.path.exists(os.path.dirname(output_file_donki_joint)):
        os.makedirs(os.path.dirname(output_file_donki_joint), exist_ok=True)

    matched_df.to_json(output_file_donki_joint, orient='records')
    # save_candidate_cmes(matched_df, output_file_candidate_cmes)

    make_elevohi_input_tracks(file_path_strudl_segmentation=file_path_strudl_segmentation,
                              output_file_strudl_tracks_with_parameters=output_file_strudl_tracks_with_parameters,
                              file_path_donki_joint=output_file_donki_joint,
                              sc='A',
                              strudl_track_dictionary=earth_pa_tracks)

