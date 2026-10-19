import functions
import numpy as np
import pandas as pd
from scipy import interpolate
import datetime
import os
from sunpy.coordinates import frames, get_horizons_coord
from sunpy.coordinates import HeliographicStonyhurst
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import constants as const
import wrappers as wrappers
import json

def get_obj_heeq_pos(pos_time, obj_name):
    """
    Get the object's positions transformed to the Heliographic Stonyhurst frame.

    Parameters
    ----------
    pos_time : array_like
        Sequence of times at which to evaluate the object's position. Typically an
        array or list of astropy.time.Time (or a time-like object accepted by
        get_horizons_coord). The same object is returned unchanged.
    obj_name : str
        Name/identifier of the solar-system object as accepted by
        get_horizons_coord.

    Returns
    -------
    tuple
        A 4-tuple (pos_time, r, lon, lat) where:
        - pos_time : same type as the input pos_time (returned unchanged)
        - r : numpy.ndarray
            Radial distances corresponding to each time in pos_time. The values are
            returned as floats.
        - lon : numpy.ndarray
            Heliographic longitudes (in degrees) for each time, returned as floats.
        - lat : numpy.ndarray
            Heliographic latitudes (in degrees) for each time, returned as floats.

    Notes
    -----
    - Longitude and latitude are explicitly converted to degrees before being
      returned. Radial distances are returned as numeric values without units.
    """
    # get object position at given time
    try:
        iter(pos_time)
    except TypeError:
        pos_time = [pos_time]

    obj_pos = get_horizons_coord(obj_name, pos_time)

    # transform object position to Heliocentric Earth Equatorial coordinates
    obj_heeq_lon = np.array([SkyCoord(pos).transform_to(HeliographicStonyhurst()).lon.to(u.rad).value for pos in obj_pos])
    obj_heeq_lat = np.array([SkyCoord(pos).transform_to(HeliographicStonyhurst()).lat.to(u.rad).value for pos in obj_pos])
    obj_heeq_r = np.array([SkyCoord(pos).transform_to(HeliographicStonyhurst()).radius.to(u.AU).value for pos in obj_pos])

    return pos_time, obj_heeq_r, obj_heeq_lon, obj_heeq_lat

def load_strudl_tracks(path, return_parameters=False):
    
    times  = []
    elongs = []

    if return_parameters:
        params = []

    #converted_strudl_dict = np.load(path, allow_pickle=True).item()
    converted_strudl_dict = json.load(open(path, 'r'))

    for cme_entry in  converted_strudl_dict:
        
        times.append([pd.to_datetime(d) for d in cme_entry["time"]])
        elongs.append(np.array(list(cme_entry["elongation"])))

        if return_parameters:

            entry_params = {'phi':cme_entry["phi"],
                           'halfwidth':cme_entry["halfwidth"],
                           'vinit':cme_entry["vinit"],
                           'L1_ist_obs':cme_entry["L1_ist_obs"],
                           'cmeID_strudl':cme_entry["cmeID_strudl"],
                           'cmeID_elevo':cme_entry["cmeID_elevo"],
                           'featureCode': cme_entry["featureCode"],
                           # the CME's own ecliptic apex longitude, so the propagation
                           # direction never has to be reconstructed from phi's sign
                           'cmeLonHeeq': cme_entry.get("cmeLonHeeq", None)}

            for flag in ['phiOutOfRange', 'phiBelowHalfwidth', 'phiBackside', 'halfwidthDegenerate']:
                entry_params[flag] = cme_entry.get(flag, None)

            params.append(entry_params)

    if return_parameters:
        return times,elongs,params
    else:
        return times,elongs


def combine_tracks(track_times,track_elongs):

    min_times    = min(np.concatenate(track_times,0))
    max_times    = max(np.concatenate(track_times,0))
    new_time_axis = functions.calculate_new_time_axis(min_times, max_times, cadence=40)

    interpolated_elongs = []
    for t in range(0,len(track_times)):
        times = track_times[t]
        x     = [(times[i] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') for i in range(0,len(times))]
        times = [(times[i]-min_times).total_seconds() for i in range(0,len(times))]
        interpf = interpolate.interp1d(x, track_elongs[t],kind='linear', fill_value="extrapolate")
        x_new = (new_time_axis - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        interpolated_elongs.append(interpf(x_new))
    

    return new_time_axis, np.array(interpolated_elongs).mean(0)

def interpolate_tracks(track_times,track_elongs,starttime=None,endtime=None):

    if starttime is None:
        min_time = min(track_times).replace(second=0, microsecond=0)
    else:
        min_time = starttime

    if endtime is None:
        max_time = max(track_times).replace(second=0, microsecond=0)
    else:
        max_time = endtime

    new_time_axis = functions.calculate_new_time_axis(min_time, max_time, cadence=40)

    x     = [(track_times[i] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s') for i in range(0,len(track_times))]

    interpf = interpolate.interp1d(x, track_elongs, kind='linear', fill_value="extrapolate")
    x_new = (new_time_axis - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    interpolated_elongs = interpf(x_new)

    return new_time_axis, np.array(interpolated_elongs)


def compute_delta(target_lon, direction):
    if abs(direction) + abs(target_lon) < np.pi:
        delta = direction - target_lon
    else:
        delta = direction - (target_lon + 2 * np.pi * np.sign(direction))

    return delta

def get_constants():
    AU_in_km = const.au.to_value('km')
    rsun_in_km = const.R_sun.to_value('km')

    return AU_in_km, rsun_in_km


def build_ELCon_inverse(d, phi, hwidth, f,
                        e_min=np.deg2rad(0.5),
                        e_max=np.deg2rad(30),
                        ngrid=3000):

    e_grid = np.linspace(e_min, e_max, ngrid)
    R_grid = functions.ELCon(e_grid, d, phi, hwidth, f)

    order = np.argsort(R_grid)
    return interpolate.interp1d(
        R_grid[order],
        e_grid[order],
        bounds_error=False,
        fill_value=np.nan
    )

def get_elevohi_t_e_track(df_combined, params_objs=None):

    AU_in_km = u.au.to(u.km)

    if params_objs is not None:
        r, sc_heeq = params_objs['STEREO-A']['r'], params_objs['STEREO-A']['lon']

    else:
        coord = get_horizons_coord('STEREO-A', df_combined['time_axis [UT]'][0])
        sc_heeq = coord.transform_to(frames.HeliographicStonyhurst)
        r = sc_heeq.radius.value

    inv_fun = build_ELCon_inverse(r*AU_in_km, np.deg2rad(df_combined['phi [°]']), np.deg2rad(df_combined['halfwidth [°]']), df_combined['f'])

    df_combined['elongation_apex [°]'] = np.rad2deg(inv_fun(df_combined['r_at_apex [km]']))
    df_combined['elongation_target_longitude [°]'] =np.rad2deg(inv_fun(df_combined['r_at_target_longitude [km]']))
    
    return df_combined

def get_mean_elevohi_results(results_array):

    results_array_final = np.zeros((np.shape(results_array)[0], 3))
    results_array_final[:,0] = np.mean(results_array, axis=1)
    results_array_final[:,1] = np.mean(results_array, axis=1) - 2*np.std(results_array, axis=1)
    results_array_final[:,2] = np.mean(results_array, axis=1) + 2*np.std(results_array, axis=1)

    return results_array_final

def run_elevohi_new(track_times, track_elongs, params, implementation_obj, params_objs):
    """
    Unified ELEvoHI.
    """
    AU_in_km, rsun_in_km = get_constants()

    phi = np.deg2rad(params["phi_manual"][0])
    halfwidth = np.deg2rad(params["halfwidth"][0])
    f = params["f"][0]

    if params["L1_ist_obs [UT]"] is None:
        return [{}]

    L1_ist_obs = datetime.datetime.strptime(params["L1_ist_obs [UT]"], "%Y-%m-%dT%H:%MZ")
    L1_isv_obs = params.get("L1_isv_obs", None)

    prediction_path = params["prediction_path"]
    vinit = params['vinit']
    cmeID_elevo = params["cmeID_elevo"]
    cmeID_strudl = params.get("cmeID_strudl", None)
    
    if params.get("starttime", None) is None:
        starttime = track_times[0].replace(second=0, microsecond=0)
    else:
        starttime = params["starttime"]

    if params.get("endtime", None) is None:
        endtime = track_times[-1].replace(second=0, microsecond=0) + datetime.timedelta(minutes=1)
    else:
        endtime = params["endtime"]

    timeaxis, elon_interp = interpolate_tracks(track_times, track_elongs, starttime=starttime, endtime=endtime)

    startcut = timeaxis.searchsorted(starttime)
    endcut = timeaxis.searchsorted(endtime) - 1

    if np.any(np.isnan(elon_interp)):
        raise ValueError("NaN values in interpolated elongation track")

    elon_rad = np.deg2rad(elon_interp)
    time = timeaxis

    sta_r, sta_lon = params_objs['STEREO-A']['r'], params_objs['STEREO-A']['lon']
    L1_r, L1_lon = params_objs['SEMB-L1']['r'], params_objs['SEMB-L1']['lon']

    # phi is the unsigned angle ELCon needs, so it cannot say which side of the Sun-STEREO-A
    # line the CME is on; take the direction from the CME's own longitude where it is
    # available, and fall back to the old reconstruction for track files that predate it
    if params.get("cmeLonHeeq", None) is not None:
        direction = np.deg2rad(params["cmeLonHeeq"])
    elif sta_lon >= 0:
        direction = sta_lon - phi
    else:
        direction = sta_lon + phi


    delta_target = compute_delta(L1_lon, direction)

    hit = implementation_obj.cme_hit(delta=delta_target, halfwidth=halfwidth)

    R_elcon = functions.ELCon(elon_rad, sta_r, phi, halfwidth, f)
    dbm_result, speed = implementation_obj.dbm_fit(
                                            time=time,
                                            distance_au=R_elcon,
                                            startcut=startcut,
                                            endcut=endcut,
                                            prediction_path=prediction_path
                                        )

    dfs_combined = []

    for fit_id, fit in dbm_result.fits.items():
        if fit.is_valid:
            gamma, winds, tinit, rinit, vinit, residual, is_valid = fit.get_parameters()
            
        else:

            dfs_combined.append({}) # no valid fit
            continue
        
        R, vdrag, time_array, tnum = implementation_obj.dbm_kinematics(
            gamma=gamma,
            winds=winds,
            tinit=tinit,
            rinit=rinit,
            vinit=vinit
        )

        pred = None
        if hit:
            pred, cme_distance_r = implementation_obj.arrival(
                R=R,
                vdrag=vdrag,
                time_array=time_array,
                tnum=tnum,
                f=f,
                halfwidth=halfwidth,
                delta_target=delta_target,
                L1_r=L1_r,
                L1_lon=L1_lon
            )

        if pred is None:
            dfs_combined.append({})
            continue

        _, _, prediction = functions.assess_prediction(
            pred,
            "L1",
            L1_ist_obs,
            L1_isv_obs
        )

        time_format = "%Y-%m-%dT%H:%MZ"
        df_combined = {
                        "cmeID_elevo": cmeID_elevo,
                        "cmeID_strudl": cmeID_strudl,
                        "featureCode": params["featureCode"],
                        "target": "L1",
                        "L1_ist_obs [UT]": datetime.datetime.strftime(L1_ist_obs, time_format),
                        "f": f,
                        "halfwidth [°]": np.rad2deg(halfwidth),
                        "phi [°]": np.rad2deg(phi),
                        "delta_target [°]": np.rad2deg(delta_target),
                        "phiOutOfRange": params.get("phiOutOfRange", None),
                        "phiBelowHalfwidth": params.get("phiBelowHalfwidth", None),
                        "phiBackside": params.get("phiBackside", None),
                        "halfwidthDegenerate": params.get("halfwidthDegenerate", None),
                        "rinit [km]": rinit,
                        "vinit [km/s]": vinit,
                        "tinit [UT]": tinit,
                        "gamma [e-7/km]": gamma * 1e7,
                        "solar wind speed [km/s]": winds,
                        "arrival_time [UT]": datetime.datetime.strftime(prediction["arrival time [UT]"], time_format),
                        "arrival_hit": hit,
                        "arrival_speed [km/s]": prediction["arrival speed [km/s]"],
                        "tnum": tnum,
                        "r_at_apex [km]": R*AU_in_km,
                        "r_at_target_longitude [km]": cme_distance_r*AU_in_km,
                        "speed [km/s]": vdrag,
                        "fit_id_dbm": fit_id,
                        "dbm_residual [R_sun]": residual,
                        "dt [h]": prediction["dt [h]"],
                        "dv [km/s]": prediction["dv [km/s]"]
                        }

        time_axis = []
        for t in df_combined['tnum']:
            time_axis.append(df_combined['tinit [UT]'] + datetime.timedelta(seconds=int(t)))
        
        df_combined['time_axis [UT]'] = np.array([datetime.datetime.strftime(tim, time_format) for tim in time_axis])
        df_combined['tinit [UT]'] = datetime.datetime.strftime(df_combined['tinit [UT]'], time_format)

        df_combined['elongation_apex_tracked [°]'] = elon_interp
        df_combined['time_apex_tracked [UT]'] = np.array([datetime.datetime.strftime(tim, time_format) for tim in timeaxis])
        df_combined['r_apex_tracked [km]'] = R_elcon*AU_in_km
        df_combined['speed_apex_tracked [km/s]'] = speed

        if implementation_obj.setup_config.get("updated_dbm_vinit_computation", None) == 'first':
            df_combined = get_elevohi_t_e_track(df_combined, params_objs)

        else:
            df_combined = get_elevohi_t_e_track(df_combined)

        dfs_combined.append(df_combined)
    
    return dfs_combined

def main_new(strudl_track_with_parameters_path, results_save_path, impl=None, config=None):

    AU_in_km, rsun_in_km = get_constants()
    all_ensembles = []
    all_ensembles_like_elevo = []
    no_pred_possible = 0

    if config is None:
        tracks_times_science, tracks_elongs_science, parameters = load_strudl_tracks(
            strudl_track_with_parameters_path,
            return_parameters=True
        )

    else:
        event_path = "data/timestep/"
        files = sorted([f for f in os.listdir(event_path) if f.endswith(".csv")])

        tracks_times_science = []
        tracks_elongs_science = []
        file_configs = []

        for file in files:
            df = pd.read_csv(event_path + file)
            tracks_times_science.append(pd.to_datetime(df["TRACK_DATE"].values))
            tracks_elongs_science.append(df["ELON"].values)
            file_configs.append(
                event_path + "config/" +
                file.replace(".csv", ".json").replace("track_", "config_")
            )


    for track_num, track in enumerate(tracks_times_science):

        prediction_path = results_save_path + f"{track_num}_"

        if config is None:
            cmeID_elevo = parameters[track_num]["cmeID_elevo"]
            cmeID_strudl = parameters[track_num]["cmeID_strudl"]

            print("Processing STRUDL ID:", cmeID_strudl,
                  "ELEvo ID:", cmeID_elevo)

            parameters_track = {
                "halfwidth": [parameters[track_num]["halfwidth"]] * 3,
                "halfwidth_step": np.nan,
                "phi_manual": [parameters[track_num]["phi"]] * 3,
                "phi_step": np.nan,
                "cmeLonHeeq": parameters[track_num].get("cmeLonHeeq", None),
                "phiOutOfRange": parameters[track_num].get("phiOutOfRange", None),
                "phiBelowHalfwidth": parameters[track_num].get("phiBelowHalfwidth", None),
                "phiBackside": parameters[track_num].get("phiBackside", None),
                "halfwidthDegenerate": parameters[track_num].get("halfwidthDegenerate", None),
                "f": [0.7, 0.7, 0.7],
                "f_step": np.nan,
                "phi_FPF_range": np.nan,
                "L1_ist_obs [UT]": parameters[track_num]["L1_ist_obs"],
                "L1_isv_obs": 999,
                "cmeID_elevo": cmeID_elevo,
                "cmeID_strudl": cmeID_strudl,
                "do_ensemble": False,
                "phi_FPF": False,
                "prediction_path": prediction_path,
                "HIobs": "A",
                "vinit": parameters[track_num]["vinit"],
                "featureCode": parameters[track_num]["featureCode"]
            }

        else:
            parameters_track = functions.load_config(file_configs[track_num])
            parameters_track["cmeID_elevo"] = file_configs[track_num].split("/")[-1].replace("config_", "").replace(".json", "")
            parameters_track["prediction_path"] = prediction_path
            parameters_track["vinit"] = None
            parameters_track["L1_ist_obs [UT]"] = datetime.datetime.strptime(parameters_track["L1_ist_obs"], "%Y-%m-%d %H:%M")
            parameters_track['starttime'] = datetime.datetime.strptime(parameters_track['starttime'], "%Y-%m-%d %H:%M")
            parameters_track['endtime'] = datetime.datetime.strptime(parameters_track['endtime'], "%Y-%m-%d %H:%M")
            parameters_track["featureCode"] = "N/A"

        # sort track times and elongs by track times
        sorted_indices = np.argsort(track)
        track = [track[i] for i in sorted_indices]
        tracks_elongs_science[track_num] = tracks_elongs_science[track_num][sorted_indices]

        sta_time, sta_r, sta_lon, _ = get_obj_heeq_pos(track[0], "STEREO-A")
        sta_r = sta_r[0]
        sta_lon = sta_lon[0]

        L1_time, L1_r, L1_lon, _ = get_obj_heeq_pos(track[0], "SEMB-L1")
        L1_r = L1_r[0]
        L1_lon = L1_lon[0]

        params_objs = {'STEREO-A': {'r': sta_r, 'lon': sta_lon},
                       'SEMB-L1': {'r': L1_r, 'lon': L1_lon}}
        
        ensemble = run_elevohi_new(
            track_times=track,
            track_elongs=tracks_elongs_science[track_num],
            params=parameters_track,
            implementation_obj=impl,
            params_objs=params_objs
        )
        valid_ensemble = False
        for ens in ensemble:
            if ens == {}:
                continue
            else:
                valid_ensemble = True
                all_ensembles.append(ens)

        if not valid_ensemble:
            print("No valid ensemble members for track number:", track_num)

            update_abs = 10

            print("Re-running ELEvoHI with different phi, half-width")

            # keep the perturbed members inside ELCon's domain: phi - 10 can go below 0 for a
            # small phi, and halfwidth - 10 can reach the value where 1 / sin(halfwidth) diverges
            phi_updated = np.array([functions.clip_phi(p) for p in
                                    np.linspace(parameters_track["phi_manual"][0]-update_abs,
                                                parameters_track["phi_manual"][0]+update_abs, 10)])
            hw_updated = np.array([functions.clip_halfwidth(h) for h in
                                   np.linspace(parameters_track["halfwidth"][0]-update_abs,
                                               parameters_track["halfwidth"][0]+update_abs, 10)])

            for phi in phi_updated:
                for hw in hw_updated:
                    parameters_track["phi_manual"] = [phi] * 3
                    parameters_track["halfwidth"] = [hw] * 3
                    ensemble = run_elevohi_new(
                        track_times=track,
                        track_elongs=tracks_elongs_science[track_num],
                        params=parameters_track,
                        implementation_obj=impl,
                        params_objs=params_objs
                    )
                    for ens_up in ensemble:
                        if ens_up != {}:
                            valid_ensemble = True
                            all_ensembles.append(ens_up)
                            break
                    if valid_ensemble:
                        break
                if valid_ensemble:
                    break


            if not valid_ensemble:
                print("No valid ensemble members for track number:", track_num, "even after parameter update")
                no_pred_possible += 1
                continue
        
        # compute elevo-like ensembles here
        mean_r_at_apex = np.zeros((len(ensemble[0]['r_at_apex [km]']), len(ensemble)))
        mean_speed = np.zeros((len(ensemble[0]['speed [km/s]']), len(ensemble)))
        mean_gamma = []
        mean_wind = []

        for i in range(len(ensemble)):
            mean_r_at_apex[:,i] = ensemble[i]['r_at_apex [km]']
            mean_speed[:,i] = ensemble[i]['speed [km/s]']
            mean_gamma.append(ensemble[i]['gamma [e-7/km]'])
            mean_wind.append(ensemble[i]['solar wind speed [km/s]'])

        mean_r_at_apex = get_mean_elevohi_results(mean_r_at_apex)
        mean_speed = get_mean_elevohi_results(mean_speed)

        delta_target = np.zeros_like(mean_r_at_apex)
        delta_target[:] = np.deg2rad(ensemble[0]['delta_target [°]'])
        distance_target, _, _, _ = functions.elevo_analytic_new(mean_r_at_apex/AU_in_km, ensemble[0]['f'], np.deg2rad(ensemble[0]['halfwidth [°]']), delta_target)
        
        time_format = "%Y-%m-%dT%H:%MZ"
        tim_axis = np.array([datetime.datetime.strptime(tim,time_format) for tim in ensemble[0]['time_axis [UT]']])
        earth_r = get_obj_heeq_pos(datetime.datetime.strptime(ensemble[0]['tinit [UT]'],time_format), "SEMB-L1")[1][0]
        results_target = functions.compute_arrival(distance_target, mean_speed, tim_axis, earth_r)
        
        
        df_combined_like_elevo = {
                        "cmeID_elevo": ensemble[0]['cmeID_elevo'],
                        "cmeID_strudl": ensemble[0]['cmeID_strudl'],
                        "featureCode": ensemble[0]['featureCode'],
                        "L1_ist_obs [UT]": ensemble[0]['L1_ist_obs [UT]'],
                        "f": ensemble[0]['f'],
                        "phi [°]": ensemble[0]['phi [°]'],
                        "halfwidth [°]": ensemble[0]['halfwidth [°]'],
                        # this is the CME-to-L1 separation, not a HEEQ longitude; the
                        # column used to be called "longitude [°]", which collided with the
                        # genuine HEEQ longitude of the same name in the ELEvo results
                        "delta_target [°]": np.round(ensemble[0]['delta_target [°]'],1),
                        "phiOutOfRange": ensemble[0].get("phiOutOfRange", None),
                        "phiBelowHalfwidth": ensemble[0].get("phiBelowHalfwidth", None),
                        "phiBackside": ensemble[0].get("phiBackside", None),
                        "halfwidthDegenerate": ensemble[0].get("halfwidthDegenerate", None),
                        "rinit [km]": ensemble[0]['rinit [km]'],
                        "vinit [km/s]": ensemble[0]['vinit [km/s]'],
                        "tinit [UT]": ensemble[0]['tinit [UT]'],
                        "gamma [e-7/km]": np.nanmean(mean_gamma),
                        "solar wind speed [km/s]": np.nanmean(mean_wind),
                        "arrival_time [UT]": datetime.datetime.strftime(results_target["arr_time_fin"][0],time_format),
                        "arrival_time_min [UT]": datetime.datetime.strftime(results_target["arr_time_err0"][0],time_format),
                        "arrival_time_max [UT]": datetime.datetime.strftime(results_target["arr_time_err1"][0],time_format),
                        "arrival_hit": results_target["arr_hit"][0],
                        "arrival_speed [km/s]": results_target["arr_speed_list"][0],
                        "arrival_speed_err [km/s]": results_target["arr_speed_err_list"][0],
                        "tnum": ensemble[0]['tnum'],
                        "time_axis [UT]": ensemble[0]['time_axis [UT]'],
                        "r_at_apex [km]": mean_r_at_apex,
                        "r_at_target_longitude [km]": distance_target*AU_in_km,
                        "speed [km/s]": mean_speed,
                        }

        if impl.setup_config.get("updated_dbm_vinit_computation", None) == 'first':
            df_combined_like_elevo = get_elevohi_t_e_track(df_combined_like_elevo, params_objs)

        else:
            df_combined_like_elevo = get_elevohi_t_e_track(df_combined_like_elevo)

        all_ensembles_like_elevo.append(df_combined_like_elevo)

    print(f'Total number of tracks: {len(tracks_times_science)}')
    print(f'Tracks with no valid fits: {no_pred_possible}')

    return all_ensembles, all_ensembles_like_elevo, no_pred_possible, track_num + 1

def make_implementation(use_baseline, setup_config):
    if use_baseline:
        return wrappers.BaselineImplementation()
    
    return wrappers.UpdatedImplementation(setup_config)

if __name__ == "__main__":
    results_save_path = 'results/'

    if not os.path.exists(results_save_path):
        os.makedirs(results_save_path)

    baselines = [False]
    track_types = ['raw', 'cleaned']
    
    raw = '../STRUDL_Tracking_Analysis/data/mean_45_2024_05_01_2025_05_01_strudl_tracks.json'
    corrected = '../STRUDL_Tracking_Analysis/data/mean_45_2024_05_01_2025_05_01_strudl_tracks_corrected.json'
    
    file_types = {'cleaned': corrected, 'raw': raw}

    for track_type in track_types:
        strudl_track_with_parameters_path = file_types[track_type]

        for bnum, use_baseline in enumerate(baselines):
            results = []

            if use_baseline:
                config = True
                setup_config = None

            else:
                config = None

                setup_config = {
                    "use_dbm_updated": True,
                    "updated_dbm_vinit_computation": 'first',
                    "allow_multiple_dbm_fits": True,
                    "use_cme_hit_function_updated": True,
                    "use_arrival_computation_updated": True,
                    "use_elevo_ensembles": False,
                    "kwargs_dbm": {"num_points": 3}
                }
            
            impl = make_implementation(use_baseline, setup_config)

            ensemble, ensemble_like_elevo, no_pred, no_total = main_new(strudl_track_with_parameters_path=strudl_track_with_parameters_path,
                results_save_path=results_save_path,
                impl=impl,
                config=config)

            # save as json
            ensemble_like_elevo_df = pd.DataFrame(ensemble_like_elevo)
            ensemble_like_elevo_df.to_json(results_save_path + 'elevohi_ensembles_results_'+track_type+'.json', orient='records', date_format='iso')

            ensemble_df = pd.DataFrame(ensemble)
            ensemble_df.to_json(results_save_path + 'elevohi_results_'+track_type+'.json', orient='records', date_format='iso')
