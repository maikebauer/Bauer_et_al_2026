import pickle
from datetime import datetime,timedelta
import matplotlib.dates as mdates
import json 
import pandas as pd 
import numpy as np
from sunpy.coordinates import frames, get_horizons_coord
import os
import re
from pred_utils import sphere2cart

def parse_space_obj_names(name):

    name_clean = name.lower()
    name_clean = re.sub(r'[^a-z0-9]', '', name_clean)

    name_num_mappings = {
        -8: ['wind'],
        -96: ['psp', 'parker', 'parkersolarprobe', 'spp', 'solarprobeplus'],
        -121: ['bepi', 'bepicolombo'],
        -144: ['solo', 'solarorbiter'],
        -234: ['sta', 'stereoa', 'stereoahead'],
        -235: ['stb', 'stereob', 'stereobehind'],
        31: ['l1', 'sembl1'],
        199: ['mercury'],
        299: ['venus'],
        399: ['earth'],
        499: ['mars'],
        599: ['jupiter'],
        699: ['saturn'],
        799: ['uranus'],
        899: ['neptune'],
    }

    name_mapping ={
        -8: 'wind',
        -96: 'psp',
        -121: 'bepi',
        -144: 'solo',
        -234: 'sta',
        -235: 'stb',
        31: 'l1',
        199: 'mercury',
        299: 'venus',
        399: 'earth',
        499: 'mars',
        599: 'jupiter',
        699: 'saturn',
        799: 'uranus',
        899: 'neptune',
    }

    name_mapped = [code for code, names in name_num_mappings.items() if name_clean in names]

    if len(name_mapped) == 1:
        return name_mapped[0], name_mapping[name_mapped[0]]
    else:
        raise ValueError(f"Unknown or ambiguous space object name: {name}")
                         
def create_positions_file(space_obj,start,stop,step='10min',save_path=''):

    start_dt = datetime.strptime(start, '%Y-%m-%d')
    stop_dt = datetime.strptime(stop, '%Y-%m-%d')

    num_seconds = (stop_dt - start_dt).total_seconds()
    step_size_seconds = int(step.split('m')[0])*60

    max_lines_jpl = 90024

    coord_dict = {}

    space_obj_code, space_obj_name = parse_space_obj_names(space_obj)

    if num_seconds/step_size_seconds > max_lines_jpl:
        num_exceed = np.ceil((num_seconds/step_size_seconds)/max_lines_jpl)
        print(f"Number of lines exceeds JPL Horizons maximum of {max_lines_jpl} by a factor of {num_exceed:.2f}.")

        obj_time = []
        obj_r = []
        obj_lon = []
        obj_lat = []
        obj_x = []
        obj_y = []
        obj_z = []
        
        # Split into multiple requests
        current_start = start_dt

        while current_start < stop_dt:
            current_stop = current_start + timedelta(seconds=(max_lines_jpl-1)*step_size_seconds)
            if current_stop > stop_dt:
                current_stop = stop_dt

            timerange = {'start':current_start, 'stop':current_stop, 'step':step}
            try:
                coord = get_horizons_coord(space_obj_code, timerange)

            except ValueError as e:
                print(f"Error retrieving data for {space_obj} from {current_start} to {current_stop}: {e}")
                calc_times = [current_start + timedelta(seconds=i*step_size_seconds) for i in range(int((current_stop - current_start).total_seconds()/step_size_seconds + 1))]
                calc_times = [datetime.strptime(calc_times[i].strftime('%Y-%m-%d %H:%M'+':00'), '%Y-%m-%d %H:%M:%S') for i in range(len(calc_times))]

                obj_time.extend(calc_times)
                obj_r.extend([np.nan]*int((current_stop - current_start).total_seconds()/step_size_seconds + 1))
                obj_lon.extend([np.nan]*int((current_stop - current_start).total_seconds()/step_size_seconds + 1))
                obj_lat.extend([np.nan]*int((current_stop - current_start).total_seconds()/step_size_seconds + 1))

                obj_x.extend([np.nan]*int((current_stop - current_start).total_seconds()/step_size_seconds + 1))
                obj_y.extend([np.nan]*int((current_stop - current_start).total_seconds()/step_size_seconds + 1))
                obj_z.extend([np.nan]*int((current_stop - current_start).total_seconds()/step_size_seconds + 1))

                current_start = current_stop + timedelta(seconds=step_size_seconds)
                continue

            # remove leap seconds because datetime is not comaptible with them
            obj_time.extend([datetime.strptime(coord[i].obstime.strftime('%Y-%m-%d %H:%M'+':00'), '%Y-%m-%d %H:%M:%S') for i in range(len(coord))])#obj_time.extend(coord.obstime.to_datetime())
            obj_r.extend(coord.radius.value)
            obj_lon.extend(np.deg2rad(coord.lon.value))
            obj_lat.extend(np.deg2rad(coord.lat.value))

            x,y,z = sphere2cart(np.array(coord.radius.value), np.deg2rad(np.array(coord.lat.value)), np.deg2rad(np.array(coord.lon.value)))
            obj_x.extend(x)
            obj_y.extend(y)
            obj_z.extend(z)

            current_start = current_stop + timedelta(seconds=step_size_seconds)

    
    else:
        timerange = {'start':start_dt, 'stop':stop_dt, 'step':step}

        try:
            coord = get_horizons_coord(space_obj_code, timerange)
            # remove leap seconds because datetime is not comaptible with them
            obj_time = [datetime.strptime(coord[i].obstime.strftime('%Y-%m-%d %H:%M')+':00', '%Y-%m-%d %H:%M:%S') for i in range(len(coord))]
            obj_r = coord.radius.value
            obj_lon = np.deg2rad(coord.lon.value)
            obj_lat = np.deg2rad(coord.lat.value)

            x,y,z = sphere2cart(np.array(coord.radius.value), np.deg2rad(np.array(coord.lat.value), np.deg2rad(np.array(coord.lon.value))))
            obj_x = x
            obj_y = y
            obj_z = z

        except ValueError as e:
            print(f"Error retrieving data for {space_obj} from {start_dt} to {stop_dt}: {e}")

            calc_times = [start_dt + timedelta(seconds=i*step_size_seconds) for i in range(int((stop_dt - start_dt).total_seconds()/step_size_seconds + 1))]
            calc_times = [datetime.strptime(calc_times[i].strftime('%Y-%m-%d %H:%M'+':00'), '%Y-%m-%d %H:%M:%S') for i in range(len(calc_times))]

            obj_time = calc_times
            obj_r = [np.nan]*int((stop_dt - start_dt).total_seconds()/step_size_seconds + 1)
            obj_lon = [np.nan]*int((stop_dt - start_dt).total_seconds()/step_size_seconds + 1)
            obj_lat = [np.nan]*int((stop_dt - start_dt).total_seconds()/step_size_seconds + 1)

            obj_x = [np.nan]*int((stop_dt - start_dt).total_seconds()/step_size_seconds + 1)
            obj_y = [np.nan]*int((stop_dt - start_dt).total_seconds()/step_size_seconds + 1)
            obj_z = [np.nan]*int((stop_dt - start_dt).total_seconds()/step_size_seconds + 1)

    
    if not os.path.exists(save_path.split('/')[0]):
        os.makedirs(save_path.split('/')[0])

    coord_dict[space_obj_name] = {'time': np.array(obj_time), 'r': np.array(obj_r), 'lon': np.array(obj_lon), 'lat': np.array(obj_lat), 'x': np.array(obj_x), 'y': np.array(obj_y), 'z': np.array(obj_z)}
    with open(save_path, 'wb') as f:
        pickle.dump(coord_dict, f)


def load_positions_jpl(data_path, date_range, sc):
    
    with open(data_path, "rb") as f:
        pos = pickle.load(f)
    
    pos_sc = pos[sc]
    time_array = pos_sc['time']
    r_array = pos_sc['r']
    lon_array = pos_sc['lon']
    lat_array = pos_sc['lat']
    x_array = pos_sc['x']
    y_array = pos_sc['y']
    z_array = pos_sc['z']

    idx = np.argwhere(np.logical_and(mdates.date2num(time_array)>=date_range[0],mdates.date2num(time_array)<=date_range[1]))

    return {'time': time_array[idx].flatten(), 'r': r_array[idx].flatten(), 'lon': lon_array[idx].flatten(), 'lat': lat_array[idx].flatten(), 'x': x_array[idx].flatten(), 'y': y_array[idx].flatten(), 'z': z_array[idx].flatten()}

def get_target_prop(name,thi):
    coord = get_horizons_coord(name, thi)
    # sc_hee = coord.transform_to(frames.HeliocentricEarthEcliptic)  #HEE
    sc_heeq = coord.transform_to(frames.HeliographicStonyhurst) #HEEQ
        
    time = sc_heeq.obstime.to_datetime()
    r = sc_heeq.radius.value
    lon = np.deg2rad(sc_heeq.lon.value)
    lat = np.deg2rad(sc_heeq.lat.value)
    return time,r,lon,lat

def load_position(data_path,date_range,spc=["l1","solo"]):
    with open(data_path+'positions_from_2010_to_2030_HEEQ_10min_rad_ed.p', "rb") as f:
        pos = pickle.load(f)
    columns = pos['l1'].dtype.names
    positions = {}
    for s in spc:
        idx = np.argwhere(np.logical_and(pos[s][columns[0]]>=date_range[0],pos[s][columns[0]]<=date_range[1]))
        positions[s] = pos[s][idx]
     
    return positions


def load_donki(results_path,dates):

    f = open(results_path)
    data = json.load(f)
    CMEs = []
    check_keys = ["time21_5", "latitude", "longitude", "halfAngle", "speed", "associatedActivityID"]

    for d in data:
        invalid_key = 0
        for key in check_keys:
            if d[key] is None:
                invalid_key = 1

        if invalid_key == 1:
            continue
        
        # if not d["hasEarthArrival"]:
        #     continue
            
        if not d["arrivalEllipsePossible"]:
            continue
        
        if (d["time21_5"]!="NaT") and (datetime.strptime(d["time21_5"],"%Y-%m-%dT%H:%MZ")>=dates[0]) and (datetime.strptime(d["time21_5"],"%Y-%m-%dT%H:%MZ")<=dates[1]):
            d["time21_5"] = datetime.strptime(d["time21_5"],"%Y-%m-%dT%H:%MZ")

            if d["rinitCorrected"] is None:
                d["rinitCorrected"] = 21.5

            CMEs.append(d)
    
        else:
            continue

    return CMEs

def load_strudl_tracks(path, return_parameters=False):
    
    times  = []
    elongs = []

    if return_parameters:
        params = []

    converted_strudl_dict = np.load(path, allow_pickle=True).item()

    for cme_key in converted_strudl_dict.keys():
        
        times.append([pd.to_datetime(d) for d in converted_strudl_dict[cme_key]["time"]])
        elongs.append(list(converted_strudl_dict[cme_key]["elongation"]))

        if return_parameters:
            phi = converted_strudl_dict[cme_key]["phi"]
            halfwidth = converted_strudl_dict[cme_key]["halfwidth"]
            L1_ist_obs = converted_strudl_dict[cme_key]["L1_ist_obs"]
            cmeID_elevo = converted_strudl_dict[cme_key]["cmeID_elevo"]

            params.append({'phi':phi, 'halfwidth':halfwidth, 'L1_ist_obs':L1_ist_obs, 'cmeID_strudl':cme_key, 'cmeID_elevo':cmeID_elevo})

    if return_parameters:
        return times,elongs,params
    else:
        return times,elongs

def load_donki_ecliptic_corrections(path):
    ecliptic_corrections = json.load(open(path, 'r'))
    return ecliptic_corrections

def apply_ecliptic_corrections(ecliptic_corrections, dat):

    feature_code = dat['featureCode']
    cme_id = dat['associatedActivityID']

    if cme_id in ecliptic_corrections:
        correction = ecliptic_corrections[cme_id][feature_code]
        dat['corrected_rinit'] = correction

    else:
        dat['corrected_rinit'] = 21.5  # default value if no correction is available

    return dat

if __name__ == "__main__":
    print(len(load_donki("./", [datetime(2024, 5, 1), datetime(2025, 5, 1)])))