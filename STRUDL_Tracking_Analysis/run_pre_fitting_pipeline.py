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
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ELEvoHI'))
import functions as elevohi_functions


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

def get_donki_strudl_match_json(file_path_strudl_segmentation, output_file_strudl_cmes, analyses_only_df, sc='A',
                                window_pad_minutes=40, threshold_fpf_minutes=60*3,
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

    if not os.path.exists(os.path.dirname(output_file_strudl_cmes)):
        os.makedirs(os.path.dirname(output_file_strudl_cmes), exist_ok=True)
        
    df_strudl.to_json(output_file_strudl_cmes, orient='records')

    if central_pa_tracks is not None and sta_radii_au is not None:
        df_strudl = helper_functions.add_fpf_fits(df_strudl, central_pa_tracks, sta_radii_au)

    final_df = helper_functions.get_matching_cmes(df_pred=df_strudl, df_analyses=analyses_only_df,
                                                  window_pad_minutes=window_pad_minutes,
                                                  threshold_fpf=threshold_fpf_minutes)

    return final_df

def make_elevohi_input_tracks(file_path_strudl_segmentation, output_file_strudl_tracks_with_parameters, file_path_donki_joint, sc='A', strudl_track_dictionary=None, output_file_exclusions=None):

    """
    `strudl_track_dictionary` is the STRUDL tracks already cut at the Earth position angle; pass
    it in to avoid re-reading the ~1 GB segmentation file and re-querying Earth's position angle
    for every CME.

    phi is the angle between the Sun-STEREO-A line and the CME propagation direction, taken
    from the *ecliptic* apex (`longitudeCorrected`) because the track is cut at the Earth
    position angle and ELCon's geometry is two-dimensional. A CME without an ecliptic
    correction does not propagate in the ecliptic, so neither ELEvoHI nor ELEvo can forecast
    it; those are excluded here and listed in `output_file_exclusions`.
    """


    os.makedirs(os.path.dirname(output_file_strudl_tracks_with_parameters), exist_ok=True)
    if strudl_track_dictionary is None:
        strudl_track_dictionary = helper_functions.convert_strudl_output(file_path_strudl_segmentation, pa_array=None, use_earth_pa=True, use_max_elon_pa=False, sc=sc)

    donki_df = pd.read_json(file_path_donki_joint)

    # sort and de-duplicate each track up front, so the STEREO-A position below is queried at
    # the track's actual first measurement rather than at whichever point happens to be stored
    # first
    tracks_sorted = {}
    for sid in strudl_track_dictionary.keys():
        sort_order = np.argsort(strudl_track_dictionary[sid]['time']).astype(int)
        time_sorted = np.array(strudl_track_dictionary[sid]['time'])[sort_order]
        elon_sorted = np.array(strudl_track_dictionary[sid]['elongation'])[sort_order]

        if len(np.unique(time_sorted)) != len(time_sorted):
            sorted_dates_unique = []
            sorted_elon_unique = []

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

        tracks_sorted[sid] = (sorted_dates_unique, sorted_elon_unique)

    track_keys = list(strudl_track_dictionary.keys())
    stereo_heeq_lons = helper_functions.get_stereo_heeq_lon([tracks_sorted[key][0][0] for key in track_keys], sc='A')
    stereo_heeq_lons_dict = {key: stereo_heeq_lons[i] for i, key in enumerate(track_keys)}

    strudl_tracks_with_parameters = []
    exclusions = []
    for sid in track_keys:
        if sid in donki_df['associatedStrudlID'].values:
            associated_rows = donki_df[donki_df['associatedStrudlID'] == sid]
            for row_ind, row in associated_rows.iterrows():

                cme_lon = row['longitudeCorrected']
                halfwidth = row['halfAngleCorrected']

                if cme_lon is None or halfwidth is None or pd.isna(cme_lon) or pd.isna(halfwidth):
                    exclusions.append({'cmeID_strudl': sid,
                                       'cmeID_elevo': row['associatedActivityID'],
                                       'featureCode': row['featureCode'],
                                       'reason': 'no_ecliptic_correction'})
                    continue

                sorted_dates_unique, sorted_elon_unique = tracks_sorted[sid]

                sta_lon = stereo_heeq_lons_dict[sid]
                donki_phi, donki_phi_signed = elevohi_functions.phi_from_longitudes(sta_lon, cme_lon)
                donki_phi = float(donki_phi)

                phi_flags = elevohi_functions.check_phi(donki_phi, halfwidth,
                                      max_elongation_deg=float(np.max(sorted_elon_unique)))

                record = {'cmeID_strudl': sid,
                          'cmeID_elevo': row['associatedActivityID'],
                          'featureCode': row['featureCode'],
                          'L1_ist_obs': row['earthArrivalTime'],
                          'phi': donki_phi,
                          'phiSignedSep': float(donki_phi_signed),
                          'cmeLonHeeq': float(cme_lon),
                          'staLonHeeq': float(sta_lon),
                          'phiRefTime': sorted_dates_unique[0],
                          'vinit': row['speed'],
                          'halfwidth': halfwidth,
                          'time_21_5Rs': row['time21_5'],
                          'time': sorted_dates_unique,
                          'elongation': sorted_elon_unique,
                          'pa': strudl_track_dictionary[sid]['pa']
                          }
                record.update(phi_flags)

                strudl_tracks_with_parameters.append(record)

    strudl_tracks_with_parameters = pd.DataFrame(strudl_tracks_with_parameters)
    strudl_tracks_with_parameters.to_json(output_file_strudl_tracks_with_parameters, orient='records', date_format='iso')

    if output_file_exclusions is None:
        output_file_exclusions = output_file_strudl_tracks_with_parameters.replace('.json', '_excluded.json')

    # pd.DataFrame(exclusions).to_json(output_file_exclusions, orient='records')

    return strudl_tracks_with_parameters


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
    (cme, time_grid, sta_coords_grid, earth_coords_grid, max_age_days, window_max_age_days,
     transit_cap_hours) = task

    extrapol = helper_functions.cme_trajectory_extrapolate(cme)
    extrapolate_rsun = extrapol[1]

    entry_time, entry_distance, early_time, late_time = helper_functions.hi1a_fov_entry_window(
        cme, time_grid, sta_coords_grid, earth_coords_grid, max_age_days=max_age_days,
        window_max_age_days=window_max_age_days)

    if entry_time is None:
        return None, None, None, None, float(extrapolate_rsun), None, None

    exit_time, exit_distance = helper_functions.hi1a_fov_exit_time(
        cme, time_grid, sta_coords_grid, entry_time,
        transit_cap_hours=transit_cap_hours, extrapol=extrapol)

    return (entry_time.strftime('%Y-%m-%dT%H:%MZ'), float(entry_distance),
            early_time.strftime('%Y-%m-%dT%H:%MZ'),
            None if late_time is None else late_time.strftime('%Y-%m-%dT%H:%MZ'),
            float(extrapolate_rsun),
            None if exit_time is None else exit_time.strftime('%Y-%m-%dT%H:%MZ'),
            None if exit_distance is None else float(exit_distance))


def get_hi1a_fov_possible(donki_df, timeframe=[], max_age_days=2.0, window_max_age_days=5.0,
                          dt_minutes=20, n_workers=None,
                          helcats_catalogue_path='./data/HCME_WP3_V06.json',
                          helcats_profile_dir='./data/HCME_WP3_V06_TE_PROFILES'):
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

    The transit is closed off at the other end by 'predictedLastHI1Atime', the last time the
    CME's leading edge is still inside the FOV, with the apex distance it has reached by then as
    'predictedLastHI1AdistanceRsun'. Entry is decided by the whole ellipsoid but the exit by its
    apex alone, since a wide CME keeps some part of itself projected into the FOV long after its
    front has swept through -- see helper_functions.hi1a_fov_exit_time. Geometry alone still
    does not end the transit for a CME heading nearly along the Sun-spacecraft line, whose apex
    never crosses the outer edge, so the exit is capped at the 95th percentile of the HELCATS
    WP3 transit times -- see helper_functions.hi1a_transit_time_cap. A CME still in the FOV at
    the cap exits at the cap.
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
        donki_df['predictedLastHI1Atime'] = None
        donki_df['predictedLastHI1AdistanceRsun'] = None
        return donki_df

    times21_5 = pd.to_datetime(donki_df.loc[valid_mask, 'time21_5'], format='%Y-%m-%dT%H:%MZ', utc=True)
    start_times = pd.to_datetime(donki_df.loc[valid_mask, 'associatedCMEstartTime'],format='%Y-%m-%dT%H:%MZ', utc=True, errors='coerce')

    # if there is no start time, the time21_5 is used as the start time
    scan_starts = start_times.fillna(times21_5).combine(times21_5, min)

    transit_cap_hours = helper_functions.hi1a_transit_time_cap(helcats_catalogue_path,
                                                               helcats_profile_dir)
    print(f"HI-1A transit time capped at {transit_cap_hours:.1f} h "
          "(95th percentile of the HELCATS WP3 STEREO-A tracks)")

    # builds time grid from the earliest scan start to the latest time21_5 + window_max_age_days,
    # with room for the capped exit scan that runs on past the latest entry
    grid_start = scan_starts.min()
    grid_end = (times21_5.max() + pd.Timedelta(days=window_max_age_days)
                + pd.Timedelta(hours=transit_cap_hours))
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
                                             + pd.Timedelta(days=window_max_age_days)
                                             + pd.Timedelta(hours=transit_cap_hours),
                                             side='right')
        window = slice(window_start, window_end)

        # save in task, use indices to get sta1 and earth cordinates
        tasks.append((cme, time_grid[window], sta_coords_grid[window], earth_coords_grid[window],
                      max_age_days, window_max_age_days, transit_cap_hours))

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
    exit_times = []
    exit_distances = []

    for valid in is_valid_row:
        (entry_time, entry_distance, early_time, late_time, extrapolate_rsun,
         exit_time, exit_distance) = (
            next(results_iter) if valid else (None, None, None, None, None, None, None))
        entry_times.append(entry_time)
        entry_distances.append(entry_distance)
        early_times.append(early_time)
        late_times.append(late_time)
        pivot_radii.append(extrapolate_rsun)
        exit_times.append(exit_time)
        exit_distances.append(exit_distance)

    donki_df['predictedFirstHI1Atime'] = entry_times
    donki_df['predictedFirstHI1AdistanceRsun'] = entry_distances
    donki_df['predictedFirstHI1AtimeEarly'] = early_times
    donki_df['predictedFirstHI1AtimeLate'] = late_times
    donki_df['predictedFirstHI1ApivotRsun'] = pivot_radii
    donki_df['predictedLastHI1Atime'] = exit_times
    donki_df['predictedLastHI1AdistanceRsun'] = exit_distances

    # the detection criterion is that some plausible speed gets the CME into the FOV, which is
    # the fastest one, so the early edge of the window is what decides it
    donki_df['couldBeInHI1AFov'] = [early_time is not None for early_time in early_times]

    transits = ((pd.to_datetime(pd.Series(exit_times), format='%Y-%m-%dT%H:%MZ', utc=True)
                 - pd.to_datetime(pd.Series(entry_times), format='%Y-%m-%dT%H:%MZ', utc=True))
                .dt.total_seconds() / 3600.0).dropna()
    if not transits.empty:
        at_cap = int((transits >= transit_cap_hours - dt_minutes / 60.0).sum())
        print(f"HI-1A transits: {len(transits)} computed, median {transits.median():.1f} h, "
              f"{at_cap} reaching the {transit_cap_hours:.1f} h cap")

    return donki_df

OVERLAP_COEFFICIENT_GRID = np.round(np.arange(0.05, 1.0, 0.05), 2)


def _nan_safe(reducer, values):
    """`reducer` over the finite entries of `values`, NaN when there are none."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]

    return np.nan if finite.size == 0 else reducer(finite)


def _as_json_float(value):
    """A plain float, or None so that to_json writes null rather than a bare NaN token."""
    value = float(value)

    return value if np.isfinite(value) else None


def _cme_front_one(task):
    """
    Worker for get_hi1a_overlaps; must stay module-level to be picklable.

    Returns the whole window as three arrays, the per-step HI1AFronts having been unpacked into
    them, so nothing defined in this module has to cross the process boundary: under the spawn
    start method a class defined here would pickle as __main__.X and fail to resolve in the
    parent. The 1600 surface points of a step never leave the worker either, each step being
    reduced to one leading-edge profile first.
    """
    cme, times, sta_coords, earth_coords = task

    extrapol = helper_functions.cme_trajectory_extrapolate(cme)
    t_extrapol, r_extrapol = extrapol

    starts = np.full(len(times), np.nan)
    widths = np.zeros(len(times))
    leading_edges = np.full((len(times), helper_functions.N_PA_BINS), np.nan, dtype=np.float32)

    for i, (t, sta, earth) in enumerate(zip(times, sta_coords, earth_coords)):
        apex_distance = r_extrapol + cme['speed'] * (t - t_extrapol).to_value(u.s) / helper_functions.RSUN_KM
        if apex_distance <= 1.0:
            continue

        front = helper_functions.cme_front_hi1a(cme, apex_distance, sta, earth)
        if front.pa_start is not None:
            starts[i] = front.pa_start
            widths[i] = front.pa_width
            leading_edges[i] = front.leading_edge

    return starts, widths, leading_edges


def get_hi1a_overlaps(donki_df, output_path, dt_minutes=20, n_workers=None,
                      coefficient_grid=OVERLAP_COEFFICIENT_GRID):
    """
    Writes, for every pair of CMEs that share time in HI-1A's FOV, how much their position angle
    extents overlap and for how long.

    Two CMEs are candidates for obscuring one another when they are in the field at the same
    time, cover the same position angles, and their fronts are close in elongation. The overlap
    in position angle is measured by the overlap coefficient, |PA intersection| / min(extent),
    rather than by intersection over union: a narrow CME inside a wide one is completely hidden
    but scores only about 0.2 on IoU.

    Covering the same position angles is not on its own enough. Two CMEs at the same position
    angle but at opposite ends of the field in elongation are two plainly separate fronts, so
    the separation of their leading edges is reported alongside the coefficient, as
    `medianElongationSeparation`. The leading edge is the largest elongation the body reaches at
    a position angle, which is the front STRUDL annotates; see `leading_edge_separation` in
    helper_functions for why the separation is a median over the shared arc rather than a
    minimum.

    The coefficient is reported as its median over the shared window. The peak would be read off
    whichever single instant suits it best, and that is usually one of the moments at the edge of
    the window where only a sliver of one CME is inside the field and is trivially contained in
    the other; a median cannot be moved by a brief sliver, so it measures how much of the time the
    two really cover the same position angles. The peak is kept alongside it as a diagnostic.

    No threshold is applied here. Each pair also carries the time it spends above each value of
    `coefficient_grid`, so any (coefficient, duration) criterion is a lookup afterwards and a
    sweep costs nothing. Durations are counted in whole grid steps, so they are resolved to
    `dt_minutes`.

    Only the pairs are written, to `output_path`. The per-CME half of the answer is returned as
    `status_df` for the caller to put on the CME table itself, because a pair table cannot say
    why a CME is missing from it: absence could mean the CME was evaluated and no other CME
    shared the field with it, or that it was never evaluated, or that it is not visible at all.
    Reading the pair table alone therefore gives the wrong denominator.
    """
    all_entries = pd.to_datetime(donki_df['predictedFirstHI1Atime'], format='%Y-%m-%dT%H:%MZ',
                                 utc=True, errors='coerce')
    all_exits = pd.to_datetime(donki_df['predictedLastHI1Atime'], format='%Y-%m-%dT%H:%MZ',
                               utc=True, errors='coerce')

    # a CME whose flank brought it into view once its apex was already past the outer edge has no
    # leading-edge transit, so there is no extent to compare
    has_window = (donki_df['couldBeInHI1AFov'].fillna(False).astype(bool)
                  & all_entries.notna() & all_exits.notna() & (all_exits > all_entries))

    cmes = donki_df[has_window].copy()
    entries = all_entries[has_window]
    exits = all_exits[has_window]

    order = np.argsort(entries.to_numpy())
    cmes = cmes.iloc[order]
    entries = entries.iloc[order]
    exits = exits.iloc[order]

    grid_index = pd.date_range(entries.min(), exits.max(), freq=f'{dt_minutes}min')
    time_grid = Time(grid_index.to_pydatetime())

    print(f"Fetching STEREO-A and Earth ephemerides for {len(time_grid)} time steps...")
    sta_coords_grid = get_horizons_coord_chunked('STEREO-A', time_grid)
    earth_coords_grid = get_horizons_coord_chunked('399', time_grid)

    tasks = []
    slices = []
    for row_id, row in cmes.iterrows():
        window_start = grid_index.searchsorted(entries.loc[row_id], side='left')
        window_end = grid_index.searchsorted(exits.loc[row_id], side='right')
        window = slice(window_start, window_end)
        slices.append((window_start, window_end))

        cme = {'time21_5': row['time21_5'], 'longitude': row['longitude'],
               'latitude': row['latitude'], 'halfAngle': row['halfAngle'],
               'speed': row['speed'],
               'associatedCMEstartTime': row.get('associatedCMEstartTime')}

        tasks.append((cme, time_grid[window], sta_coords_grid[window], earth_coords_grid[window]))

    print(f"Measuring the HI-1A position angle extent of {len(tasks)} CMEs on "
          f"{n_workers or os.cpu_count()} workers...")
    extents = []
    if tasks:
        with ProcessPoolExecutor(max_workers=n_workers,
                                 mp_context=multiprocessing.get_context('spawn')) as executor:
            for i, result in enumerate(executor.map(_cme_front_one, tasks, chunksize=4), start=1):
                extents.append(result)
                if i % 200 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} done")

    activity_ids = cmes['associatedActivityID'].tolist()
    rows = []

    for a in range(len(tasks)):
        start_a, end_a = slices[a]
        for b in range(a + 1, len(tasks)):
            start_b, end_b = slices[b]

            # the CMEs are sorted by entry, so once b starts after a ends no later b can overlap
            if start_b >= end_a:
                break

            first = max(start_a, start_b)
            last = min(end_a, end_b)
            if last <= first:
                continue

            extent_a = extents[a]
            extent_b = extents[b]

            coefficients = []
            widths_a = []
            widths_b = []
            separations = []
            shared_bins = []
            elons_a = []
            elons_b = []
            for step in range(first, last):
                width_a = extent_a[1][step - start_a]
                width_b = extent_b[1][step - start_b]
                widths_a.append(width_a)
                widths_b.append(width_b)

                # taken before the zero-width test below so the per-step lists stay the same
                # length as coefficients; a CME with no extent has an all-NaN profile anyway
                separation, n_bins, elon_a, elon_b = helper_functions.leading_edge_separation(
                    extent_a[2][step - start_a], extent_b[2][step - start_b])
                separations.append(separation)
                shared_bins.append(n_bins)
                elons_a.append(elon_a)
                elons_b.append(elon_b)

                smaller = min(width_a, width_b)
                if smaller <= 0:
                    coefficients.append(0.0)
                    continue

                overlap = helper_functions.circular_overlap(
                    (extent_a[0][step - start_a], width_a),
                    (extent_b[0][step - start_b], width_b))
                coefficients.append(overlap / smaller)

            coefficients = np.array(coefficients)
            peak = int(np.argmax(coefficients))

            rows.append({
                'cmeIdA': activity_ids[a],
                'cmeIdB': activity_ids[b],
                'sharedWindowStart': grid_index[first].strftime('%Y-%m-%dT%H:%MZ'),
                'sharedWindowEnd': grid_index[last - 1].strftime('%Y-%m-%dT%H:%MZ'),
                'sharedWindowMinutes': float(len(coefficients) * dt_minutes),
                'medianOverlapCoefficient': float(np.median(coefficients)),
                'peakOverlapCoefficient': float(coefficients[peak]),
                'peakOverlapTime': grid_index[first + peak].strftime('%Y-%m-%dT%H:%MZ'),
                'paExtentAmedian': float(np.median(widths_a)),
                'paExtentBmedian': float(np.median(widths_b)),
                'medianElongationSeparation': _as_json_float(_nan_safe(np.nanmedian, separations)),
                'minElongationSeparation': _as_json_float(_nan_safe(np.nanmin, separations)),
                'elonLEAmedian': _as_json_float(_nan_safe(np.nanmedian, elons_a)),
                'elonLEBmedian': _as_json_float(_nan_safe(np.nanmedian, elons_b)),
                'sharedPaBinsMedian': float(np.median(shared_bins)),
                'elongationSeparationSteps': int(np.isfinite(separations).sum()),
                'minutesAboveCoefficient': {
                    f'{value:.2f}': float((coefficients >= value).sum() * dt_minutes)
                    for value in coefficient_grid},
            })

    overlaps_df = pd.DataFrame(rows)

    if len(overlaps_df):
        pair_counts = pd.Series(overlaps_df['cmeIdA'].tolist()
                                + overlaps_df['cmeIdB'].tolist()).value_counts()
    else:
        pair_counts = pd.Series(dtype=int)

    evaluated_ids = set(cmes['associatedActivityID'])
    visible = donki_df[donki_df['couldBeInHI1AFov'].fillna(False).astype(bool)]

    status_rows = []
    for row_id, row in visible.iterrows():
        entry, exit_ = all_entries.loc[row_id], all_exits.loc[row_id]

        if pd.isna(entry) or pd.isna(exit_):
            excluded_reason = 'noTransitWindow'
        elif exit_ <= entry:
            excluded_reason = 'zeroLengthTransit'
        else:
            excluded_reason = None

        cme_id = row['associatedActivityID']
        evaluated = cme_id in evaluated_ids
        n_pairs = int(pair_counts.get(cme_id, 0))

        status_rows.append({
            'associatedActivityID': cme_id,
            'overlapEvaluated': evaluated,
            'overlapExcludedReason': excluded_reason,
            'nOverlapCandidatePairs': n_pairs,
            'overlapIsolated': bool(evaluated and n_pairs == 0),
        })

    status_df = pd.DataFrame(status_rows)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    overlaps_df.to_json(output_path, orient='records')

    if len(overlaps_df):
        print(f"HI-1A overlaps: {len(overlaps_df)} pairs share time in the FOV, "
              f"median overlap coefficient {overlaps_df['medianOverlapCoefficient'].median():.2f}")
    print(f"HI-1A overlap status: {len(status_df)} visible CMEs, "
          f"{int((~status_df['overlapEvaluated']).sum())} excluded, "
          f"{int(status_df['overlapIsolated'].sum())} evaluated but sharing the field with no "
          "other CME")

    return overlaps_df, status_df


def add_candidate_flag(donki_df):
    """
    Flags every candidate CME in `donki_df` with an 'isCandidate' column. A candidate is a CME
    that was matched to a STRUDL track, or one that arrived at Earth and whose corrected arrival
    ellipse could have covered Earth. These are the CMEs the manual tracking tool works through.

    The flag replaces what used to be a separate file of the selected rows. That file repeated
    the whole table for a subset of it, and serialising the subset on its own silently changed
    dtypes -- longitude, latitude and halfAngle came back as integers where the full table gives
    floats.
    """

    matched_strudl = donki_df['associatedStrudlID'].notna()
    earth_arrival = (donki_df['hasEarthArrival'].fillna(False).astype(bool)
                     & donki_df['arrivalEllipsePossibleCorrected'].fillna(False).astype(bool))

    donki_df['isCandidate'] = matched_strudl | earth_arrival
    return donki_df

if __name__ == "__main__":

    file_path_strudl_segmentation = '../STRUDL/segmentation_dictionary_mean_45_2024_05_01_2025_05_01.npy'

    output_file_strudl_cmes = './data/STRUDL_cmes.json'

    input_file_donki_cmes = './data/DONKI_cmes.json'
    input_file_donki_ips = './data/DONKI_ips.json'

    output_file_donki_joint = './data/DONKI_cme_analyses_matched.json'

    output_file_strudl_tracks_with_parameters = './data/mean_45_2024_05_01_2025_05_01_strudl_tracks.json'
    output_file_overlaps = './data/DONKI_cme_hi1_overlaps.json'

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
                                             output_file_strudl_cmes=output_file_strudl_cmes,
                                             analyses_only_df=corrected_df,
                                             strudl_dict=strudl_dict,
                                             central_pa_tracks=central_pa_tracks,
                                             sta_radii_au=sta_radii_au
                                             )

    if not os.path.exists(os.path.dirname(output_file_donki_joint)):
        os.makedirs(os.path.dirname(output_file_donki_joint), exist_ok=True)

    # the overlap status becomes columns on the CME table, so the overlaps have to be computed
    # before it is written; make_elevohi_input_tracks reads that file back from disk, so it has
    # to stay after the write
    _, overlap_status_df = get_hi1a_overlaps(matched_df, output_file_overlaps)

    matched_df = matched_df.merge(overlap_status_df, on='associatedActivityID', how='left')
    matched_df['overlapEvaluated'] = matched_df['overlapEvaluated'].fillna(False).astype(bool)
    matched_df['overlapIsolated'] = matched_df['overlapIsolated'].fillna(False).astype(bool)
    matched_df['nOverlapCandidatePairs'] = matched_df['nOverlapCandidatePairs'].fillna(0).astype(int)
    matched_df['overlapExcludedReason'] = matched_df['overlapExcludedReason'].where(
        matched_df['associatedActivityID'].isin(overlap_status_df['associatedActivityID']),
        'notVisibleInHI1A')

    matched_df = add_candidate_flag(matched_df)

    matched_df.to_json(output_file_donki_joint, orient='records')

    make_elevohi_input_tracks(file_path_strudl_segmentation=file_path_strudl_segmentation,
                              output_file_strudl_tracks_with_parameters=output_file_strudl_tracks_with_parameters,
                              file_path_donki_joint=output_file_donki_joint,
                              sc='A',
                              strudl_track_dictionary=earth_pa_tracks)

