import numpy as np
import pandas as pd
import datetime
import glob
import os
from helper_functions import get_stereo_heeq_lon
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ELEvoHI'))
import functions as elevohi_functions


def get_corrected_dict_combined(corrected_files):
    """
    Read the hand-corrected tracks, one record per file, sorted in time.

    The tracking tool writes a file per corrected track; its name carries the CME ids, which
    are repeated in every row, so the first row is enough to identify the track.
    """

    corrected_dict = []
    for track_file in corrected_files:
        df = pd.read_csv(track_file)

        # a track finished without any clicked points marks a CME as reviewed but not
        # trackable; it carries no ids or measurements, so it stays out of the output
        if df.empty:
            print(f"skipping empty corrected track: {track_file}")
            continue

        times = pd.to_datetime(df['time'])
        sort_order = np.argsort(times.values).astype(int)
        cme_id_strudl = df['cmeID_strudl'].iloc[0]

        corrected_dict.append({
            'cmeID_strudl': cme_id_strudl if pd.notna(cme_id_strudl) else None,
            'cmeID_elevo': df['cmeID_elevo'].iloc[0],
            'featureCode': df['featureCode'].iloc[0],
            'time': [t.to_pydatetime() for t in times.iloc[sort_order]],
            'elongation': np.asarray(df['elongation'])[sort_order].tolist(),
        })

    return corrected_dict


def build_manual_track_records(corrected_dict, donki_df, strudl_tracks_by_key, sc='A'):
    """
    Attach the ELEvoHI input parameters to every hand-corrected track.

    The parameters come from the matched DONKI analyses rather than from the ELEvo results.
    ELEvo is a forecast product that has already dropped every CME which cannot geometrically
    arrive, so reading them from there silently reduced the manual tracks to the arriving
    subset. Whether a CME arrives is for the forecast codes to decide, not for the file that
    feeds them.

    phi is re-referenced to each corrected track's own first measurement, because phi depends
    on the track only through the time at which STEREO-A's longitude is sampled. Those
    longitudes are fetched in one batched Horizons query.
    """

    if not corrected_dict:
        return []

    donki_by_key = donki_df.set_index(['associatedActivityID', 'featureCode'])

    sta_lons = get_stereo_heeq_lon([entry['time'][0] for entry in corrected_dict], sc=sc)

    records = []
    for entry, sta_lon in zip(corrected_dict, sta_lons):
        key = (entry['cmeID_elevo'], entry['featureCode'])

        if key not in donki_by_key.index:
            print(f"no DONKI analysis for {key}, skipping")
            continue

        row = donki_by_key.loc[key]

        cme_lon = row['longitudeCorrected']
        halfwidth = row['halfAngleCorrected']

        # a CME without an ecliptic correction does not propagate in the ecliptic, so neither
        # ELEvoHI nor ELEvo can forecast it; this is the cut the automatic tracks already get
        if cme_lon is None or halfwidth is None or pd.isna(cme_lon) or pd.isna(halfwidth):
            continue

        phi, phi_signed = elevohi_functions.phi_from_longitudes(sta_lon, cme_lon)
        phi = float(phi)

        phi_flags = elevohi_functions.check_phi(phi, halfwidth,
                                                max_elongation_deg=float(np.max(entry['elongation'])))

        # pa is the position angle the automatic track was cut at; a manual track has no
        # position angle of its own, so it is carried over where STRUDL found the same CME
        strudl_counterpart = strudl_tracks_by_key.get(key)

        record = {'cmeID_strudl': entry['cmeID_strudl'],
                  'cmeID_elevo': entry['cmeID_elevo'],
                  'featureCode': entry['featureCode'],
                  'L1_ist_obs': row['earthArrivalTime'],
                  'phi': phi,
                  'phiSignedSep': float(phi_signed),
                  'cmeLonHeeq': float(cme_lon),
                  'staLonHeeq': float(sta_lon),
                  'phiRefTime': entry['time'][0],
                  'vinit': row['speed'],
                  'halfwidth': halfwidth,
                  'time_21_5Rs': row['time21_5'],
                  'time': format_times(entry['time']),
                  'elongation': entry['elongation'],
                  'pa': strudl_counterpart['pa'] if strudl_counterpart is not None else None
                  }
        record.update(phi_flags)

        records.append(record)

    return records


def build_paired_track_records(corrected_dict, strudl_tracks_by_key):
    """
    Pair every hand-corrected track with the automatic STRUDL track for the same CME.

    This is the manual-versus-automatic comparison set, so it carries no forecast parameters
    and is gated only on a corrected track existing: whether a CME can arrive, or did arrive,
    must not decide which tracks are available to compare. A CME STRUDL never detected, or
    whose front never reached Earth's position angle, is kept with no automatic counterpart.
    strudlTrackAvailable says an automatic track exists to compare against, which is narrower
    than STRUDL having detected the CME: cmeID_strudl is set for 135 of the tracks but only 80
    of those survive the ecliptic-correction and Earth-position-angle cuts.
    """

    paired = []
    for entry in corrected_dict:
        key = (entry['cmeID_elevo'], entry['featureCode'])
        strudl = strudl_tracks_by_key.get(key)

        paired.append({
            'cmeID_elevo': entry['cmeID_elevo'],
            'cmeID_strudl': entry['cmeID_strudl'],
            'featureCode': entry['featureCode'],
            'strudlTrackAvailable': strudl is not None,
            'time_manual': format_times(entry['time']),
            'elongation_manual': entry['elongation'],
            'time_strudl': strudl['time'] if strudl is not None else None,
            'elongation_strudl': strudl['elongation'] if strudl is not None else None
        })

    return paired


def format_times(times):
    """
    Format track times the way the downstream readers parse them.

    make_paper_plots reads the corrected tracks with this literal format and silently yields
    NaT on anything else, so the formatting is done here rather than left to to_json.
    """

    return [datetime.datetime.strftime(t, '%Y-%m-%dT%H:%M:%S.%f') for t in times]


def create_new_file_combined(path_retrack_strudl, path_donki_joint, path_paired_tracks):

    corrected_tracks_path = 'corrected_tracks_new'
    corrected_tracks = sorted(glob.glob(f"{corrected_tracks_path}/*.csv"))

    corrected_dict = get_corrected_dict_combined(corrected_tracks)

    donki_df = pd.read_json(path_donki_joint)

    strudl_tracks = json.load(open(path_retrack_strudl, 'r'))
    strudl_tracks_by_key = {(track['cmeID_elevo'], track['featureCode']): track for track in strudl_tracks}

    manual_records = build_manual_track_records(corrected_dict, donki_df, strudl_tracks_by_key)
    pd.DataFrame(manual_records).to_json(path_retrack_strudl.replace('.json', '_corrected.json'),
                                         orient='records', date_format='iso')

    paired_records = build_paired_track_records(corrected_dict, strudl_tracks_by_key)
    pd.DataFrame(paired_records).to_json(path_paired_tracks, orient='records', date_format='iso')

    n_detected = sum(record['strudlTrackAvailable'] for record in paired_records)
    print(f"{len(corrected_dict)} corrected tracks -> {len(manual_records)} with ELEvoHI parameters "
          f"({len(corrected_dict) - len(manual_records)} without an ecliptic correction)")
    print(f"{len(paired_records)} paired tracks -> {n_detected} with an automatic STRUDL counterpart")


if __name__ == "__main__":

    path_retrack_strudl = 'data/mean_45_2024_05_01_2025_05_01_strudl_tracks.json'
    path_donki_joint = 'data/DONKI_cme_analyses_matched.json'
    path_paired_tracks = 'data/manual_vs_strudl_tracks.json'

    create_new_file_combined(path_retrack_strudl, path_donki_joint, path_paired_tracks)
