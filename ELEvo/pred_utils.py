import numpy as np 
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import astropy.units as u
from sunpy.time import parse_time
from numba import njit
import pandas as pd
from sunpy.coordinates import frames, get_horizons_coord
from scipy.interpolate import interp1d
from scipy.stats import skewnorm
import logging

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
def cart2sphere(x,y,z):
    """
    Converts Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).

    Parameters
    ----------
    x : float or array-like
        The x-coordinate(s) in Cartesian space.
    y : float or array-like
        The y-coordinate(s) in Cartesian space.
    z : float or array-like
        The z-coordinate(s) in Cartesian space.

    Returns
    -------
    r : float or array-like
        The radial distance from the origin.
    theta : float or array-like
        The polar angle (inclination) in radians, measured from the z-axis.
    phi : float or array-like
        The azimuthal angle in radians, measured from the x-axis in the x-y plane.

    Notes
    -----
    - The output angles are in radians.
    """
    r = np.sqrt(x**2+ y**2 + z**2)           
    theta = np.arctan2(z,np.sqrt(x**2+ y**2))
    phi = np.arctan2(y,x)                    
    return r, theta, phi


def sphere2cart(r,theta,phi):
    """
    Converts spherical coordinates (r, theta, phi) to cartesian coordinates (x, y, z).

    Parameters
    ----------
    r : float or array-like
        The radial distance from the origin.
    theta : float or array-like
        The polar angle (inclination) in radians, measured from the z-axis.
    phi : float or array-like
        The azimuthal angle in radians, measured from the x-axis in the x-y plane.

    Returns
    -------
    x : float or array-like
        The x-coordinate(s) in Cartesian space.
    y : float or array-like
        The y-coordinate(s) in Cartesian space.
    z : float or array-like
        The z-coordinate(s) in Cartesian space.

    Notes
    -----
    - The input angles must be given in radians.
    """

    x = r*np.cos(theta)*np.cos(phi)
    y = r*np.cos(theta)*np.sin(phi)
    z = r*np.sin(theta)
                  
    return x, y, z

@njit(fastmath=False, parallel=True)
def compute_cme_ensemble(gamma, ambient_wind, speed_ensemble, timesteps, distance0):
    """
    Compute CME ensemble propagation (r and v) with Numba acceleration.

    Parameters
    ----------
    gamma : 1D array (n_ensemble)
    ambient_wind : 1D array (n_ensemble)
    speed_ensemble : 1D array (n_ensemble)
    timesteps : 1D array (kindays_in_min)
    distance0 : float

    Returns
    -------
    cme_r_ensemble : 2D array [kindays_in_min, n_ensemble]
    cme_v_ensemble : 2D array [kindays_in_min, n_ensemble]
    """

    n_ens = len(speed_ensemble)
    accsign = np.ones(n_ens)
    accsign[speed_ensemble < ambient_wind] = -1.

    cme_r_ensemble = (accsign / (gamma * 1e-7)) * np.log(1 + (accsign * (gamma * 1e-7) * ((speed_ensemble - ambient_wind) * timesteps))) + ambient_wind * timesteps + distance0
    cme_v_ensemble = (speed_ensemble - ambient_wind) / (1 + (accsign * (gamma * 1e-7) * (speed_ensemble - ambient_wind) * timesteps)) + ambient_wind

    return cme_r_ensemble, cme_v_ensemble

def compute_arrival(cme_r, cme_v, time_array, target_r):
    """
    Compute arrival time and speed at a target distance given CME distance and speed profiles.

    Parameters
    ----------
    cme_r : numpy.ndarray
        Array of CME radial distances (AU). Shape can be (N,) or (N, 3) for mean and uncertainty bounds.
    cme_v : numpy.ndarray
        Array of CME speeds (km/s). Shape must match cme_r.
    time_array : list of datetime
        List of datetime objects corresponding to the CME distance and speed profiles.
    target_r : float
        Target radial distance (AU) for arrival time calculation. 
    
    Returns
    -------
    dict
        A dictionary containing:
        - "arr_time_fin": List of estimated arrival times (datetime).
        - "arr_time_err0": List of lower bounds of arrival time uncertainty (datetime).
        - "arr_time_err1": List of upper bounds of arrival time uncertainty (datetime).
        - "arr_time_err_list": List of arrival time uncertainties in hours (float).
        - "arr_hit": List indicating if CME hits the target (1.0 for hit).
        - "arr_speed_list": List of estimated arrival speeds (km/s).
        - "arr_speed_err_list": List of arrival speed uncertainties (km/s).
    """


    # make sure that arrays have same shape when passed to function
    if cme_r.ndim != cme_v.ndim:
        raise ValueError("Input arrays must have the same number of dimensions.")

    index_hit = np.nanargmin(np.abs(cme_r - target_r), axis=0)    
    arr_time = np.array(time_array)[index_hit]

    if cme_r.ndim == 1:
        arr_speed_mean = [cme_v[index_hit]]
        err_arr_speed = 0.0
        err_arr_time = 0.0

        arr_time_mean = [arr_time]

    else:
        arr_speed_mean = [cme_v[:, 0][index_hit[0]]]
        err_arr_speed = (cme_v[:, 2][index_hit[2]] - cme_v[:, 1][index_hit[1]])/2
        err_arr_time = ((arr_time[1] - arr_time[2]).total_seconds() / 3600.0)/2

        arr_time_mean = [arr_time[0]]

    arr_time_err_lower = [arr_time_mean[0] - timedelta(hours=err_arr_time)]
    arr_time_err_upper = [arr_time_mean[0] + timedelta(hours=err_arr_time)]

    err_arr_time = [err_arr_time]
    err_arr_speed = [err_arr_speed]

    return {
        "arr_time_fin": arr_time_mean,
        "arr_time_err0": arr_time_err_lower,
        "arr_time_err1": arr_time_err_upper,
        "arr_time_err_list": err_arr_time,
        "arr_hit": [1.0],
        "arr_speed_list": arr_speed_mean,
        "arr_speed_err_list": err_arr_speed,
    }

def elevo_analytic(R, f, halfwidth, delta):
    """
    Calculate the distance from the Sun to a point along an ellipse in the ecliptic plane, which 
    describes the front of a solar transient. This point is specified by the angle delta 
    between the ellipse apex and the in situ spacecraft. The ellipse is specified by 
    the distance of the apex (R), the half width in degrees in heliospheric longitude, and the aspect ratio.
    One of the ellipse main axes is oriented along the propagation direction, the 
    other perpendicular to it. The aspect ratio can take any values, but physical 
    ones suitable for CMEs cluster most likely around 1.3 +/ 0.2.
    
    Parameters: 
        R (numpy.ndarray): Heliocentric distance of CME apex in AU
        f (float): Ellipse aspect ratio (a/b)
        halfwidth (float): Halfwidth of the ellipse
        delta (float): Angular separation of CME apex and in situ s/c

    Returns:
        distance_earth (numpy.ndarray): Distance from Sun to point along ellipse at angle delta in AU
        cme_a (numpy.ndarray): Semi-major axis of the ellipse in AU
        cme_b (numpy.ndarray): Semi-minor axis of the ellipse in AU
        cme_c (numpy.ndarray): Distance from Sun to center of ellipse in AU
    """

    theta = np.arctan(f**2 * np.tan(halfwidth))
    omega = np.sqrt(np.cos(theta)**2 * (f**2 - 1) + 1)   

    cme_b = R * omega * np.sin(halfwidth) / (np.cos(halfwidth - theta) + omega * np.sin(halfwidth))    
    cme_a = cme_b / f
    cme_c = R - cme_b
        
    root = np.sin(delta)**2 * f**2 * (cme_b**2 - cme_c**2) + np.cos(delta)**2 * cme_b**2
    distance_earth = (cme_c * np.cos(delta) + np.sqrt(root)) / (np.sin(delta)**2 * f**2 + np.cos(delta)**2) #distance from SUN in AU for given point on ellipse

    return distance_earth, cme_a, cme_b, cme_c

def does_cme_hit(delta,halfwidth, tol=0.0):
    tol_rad = np.deg2rad(tol)

    return np.abs(delta) < (halfwidth + tol_rad)

def ELCon(elon, d, phi, hwidth, f):
    """ ELlipse Conversion method (Rollett et al., 2016)

    converts HI time-elongation measurements to time-distance.
    """
    p = phi
    hw = hwidth
    e = elon
    R_ell = np.zeros(np.size(elon))

    for i in range(len(elon)):
        
        beta1 = np.pi - e[i] - p

        if beta1 > np.pi/2.:
            beta1 = e[i] + p

        theta = np.arctan(f**2 * np.tan(beta1))
        w = np.sqrt((np.cos(theta)**2) * (f**2 - 1) + 1)

        thetas = np.arctan(f**2 * np.tan(hw))
        ws = np.sqrt((np.cos(thetas)**2) * (f**2 - 1) + 1)

        X = ((np.cos(hw-thetas)/np.sin(hw)) + ws)**(-1) * ((np.cos(e[i]+p+theta)/(w*np.sin(e[i]+p))) + 1)

        if (np.pi-e[i]-p) > np.pi/2:
            X = ((np.cos(hw-thetas)/np.sin(hw))+ws)**(-1) * ((-np.sin(np.pi/2+theta-e[i]-p)/(w*np.sin(e[i]+p))) + 1)

        Y = (d * np.sin(e[i]))/(np.sin(e[i] + p))

        R_ell[i] = Y/(1 - ws*X)
        
    return R_ell

def build_ELCon_inverse(d, phi, hwidth, f,
                        e_min=np.deg2rad(0.5),
                        e_max=np.deg2rad(30),
                        ngrid=3000):

    e_grid = np.linspace(e_min, e_max, ngrid)
    R_grid = ELCon(e_grid, d, phi, hwidth, f)

    order = np.argsort(R_grid)
    return interp1d(
        R_grid[order],
        e_grid[order],
        bounds_error=False,
        fill_value=np.nan
    )

def get_elevo_t_e_track(df_combined):

    AU_in_km = u.au.to(u.km)

    coord = get_horizons_coord('STEREO-A', df_combined['time_axis [UT]'][0])
    sc_heeq = coord.transform_to(frames.HeliographicStonyhurst)
    r = sc_heeq.radius.value

    inv_fun = build_ELCon_inverse(r*AU_in_km, np.deg2rad(df_combined['phi [°]']), np.deg2rad(df_combined['halfwidth [°]']), df_combined['f'])

    df_combined['elongation_apex [°]'] = np.rad2deg(inv_fun(df_combined['r_at_apex [km]']))
    df_combined['elongation_target_longitude [°]'] =np.rad2deg(inv_fun(df_combined['r_at_target_longitude [km]']))
    
    return df_combined

def Prediction_ELEvo(time21_5, latitude, longitudeCorrected, halfAngleCorrected, speed, associatedActivityID, featureCode, positions, rinitCorrected, arrivalEllipsePossibleCorrected, earthArrivalTime, seed_value=None, **kwargs):
    print(associatedActivityID)
    AU_in_km = u.au.to(u.km)

    if seed_value is None:
        seed_value = datetime.now().year + datetime.now().month + datetime.now().day + datetime.now().hour + datetime.now().minute + datetime.now().second
    
    if rinitCorrected == 0 or rinitCorrected == -0 or rinitCorrected is None or halfAngleCorrected == 0 or halfAngleCorrected == -0 or halfAngleCorrected is None or arrivalEllipsePossibleCorrected == 0 or longitudeCorrected == 0 or longitudeCorrected == -0 or longitudeCorrected is None:
        #print(f'CME {associatedActivityID} does not hit (high latitude)')
        results_earth = {
            "arr_time_fin": [np.nan],
            "arr_time_err0": [np.nan],
            "arr_time_err1": [np.nan],
            "arr_time_err_list": [np.nan],
            "arr_id": [np.nan],
            "arr_hit": [np.nan],
            "arr_speed_list": [np.nan],
            "arr_speed_err_list": [np.nan],
        }

        arrival_for_txt = []
        df_combined = {}
        return df_combined
    
    # if associatedActivityID != '2024-07-26T15:23:00-CME-001':
    #     df_combined = {}
    #     return df_combined    

    distance0 = rinitCorrected*u.solRad.to(u.km) #21.5*u.solRad.to(u.km)
    t0 = time21_5

    ambient_wind_init = 400.
    gamma_init = 0.1
    kindays = 7
    n_ensemble = 1000
    halfwidth = np.deg2rad(halfAngleCorrected)
    res_in_min = 10
    f = 0.7
    kindays_in_min = int(kindays*24*60/res_in_min)

    earth = positions["l1"]
    earth_time = mdates.date2num(earth["time"])
    earth_lon = earth["lon"]
    #earth_lat = earth["lat"]
    earth_r = earth["r"]


    ### Just doing earth for now, if needed create generic function and call it with spacecraft pos array

    dct = earth_time - mdates.date2num(time21_5)
    earth_ind = np.argmin(np.abs(dct))

    if np.abs(np.deg2rad(longitudeCorrected)) + np.abs(earth_lon[earth_ind]) > np.pi and np.sign(np.deg2rad(longitudeCorrected)) != np.sign(earth_lon[earth_ind]):
        delta_earth = np.deg2rad(longitudeCorrected) - (earth_lon[earth_ind] + 2 * np.pi * np.sign(np.deg2rad(longitudeCorrected)))
    else:
        delta_earth = np.deg2rad(longitudeCorrected) - earth_lon[earth_ind]

    #times for each event kinematic
    time1=[]
    tstart1=time21_5
    tend1=tstart1+timedelta(days=kindays)
    #make 30 min datetimes
    while tstart1 < tend1:

        time1.append(tstart1)  
        tstart1 += timedelta(minutes=res_in_min)    


    # #make kinematics
    #timestep=np.zeros([kindays_in_min,n_ensemble])
    cme_r=np.zeros([kindays_in_min, 3])
    cme_v=np.zeros([kindays_in_min, 3])
    cme_lon=np.ones(kindays_in_min)*longitudeCorrected
    cme_lat=np.ones(kindays_in_min)*latitude
    cme_id=np.chararray(kindays_in_min, itemsize=27)
    cme_id[:]=associatedActivityID
    cme_r_ensemble=np.zeros([kindays_in_min, n_ensemble])
    cme_v_ensemble=np.zeros([kindays_in_min, n_ensemble])
    

    cme_delta=delta_earth*np.ones([kindays_in_min,3])

    cme_hit = does_cme_hit(cme_delta, halfwidth)

    distance_earth = np.empty([kindays_in_min,3])
    distance_solo = np.empty([kindays_in_min,3])
    distance_sta = np.empty([kindays_in_min,3])
    distance_earth[:] = np.nan
    distance_solo[:] = np.nan
    distance_sta[:] = np.nan
        
    kindays_in_min = int(kindays*24*60/res_in_min)

    rng = np.random.default_rng(seed_value)

    speed_delta = 115
    speed_ensemble = rng.normal(loc=speed, scale=speed_delta/2, size=n_ensemble) # CME speed, normally distributed around v_0 with a standard deviation of speed_delta

    gamma_values = np.array([7.5, 22.5, 7.5, 15, 7.5, 11, 7.5, 7.5, 3.5, 3.5, 3.5, 3.5]) # Sum = 100
    gamma_values = gamma_values / np.sum(gamma_values)  # Normalize to sum to 1

    gamma_bin_edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])

    gamma_bin_centers = (gamma_bin_edges[:-1] + gamma_bin_edges[1:]) / 2

    synthetic_gamma_values = rng.choice(gamma_bin_centers, size=n_ensemble, p=gamma_values)

    gamma_shape, gamma_loc, gamma_scale = skewnorm.fit(synthetic_gamma_values)

    sampled_gamma = skewnorm.rvs(gamma_shape, loc=gamma_loc, scale=gamma_scale, size=n_ensemble)

    # remove all values below 0.05 and above 2.4
    clipped_sampled_gamma = sampled_gamma[(sampled_gamma >= 0.05) & (sampled_gamma <= 2.4)]

    # check if clipped_sampled_gamma contains enough values, if not, resample and add more values
    while len(clipped_sampled_gamma) < n_ensemble:
        additional_sample = skewnorm.rvs(gamma_shape, loc=gamma_loc, scale=gamma_scale, size=n_ensemble)
        additional_clipped_sample = additional_sample[(additional_sample >= 0.05) & (additional_sample <= 2.4)]
        clipped_sampled_gamma = np.concatenate([clipped_sampled_gamma, additional_clipped_sample])
        clipped_sampled_gamma = clipped_sampled_gamma[:n_ensemble]

    w_values = np.array([0, 0, 27.5, 26.5, 22.5, 18, 5, 0.5, 0]) # Sum = 100
    w_values = w_values / np.sum(w_values)  # Normalize to sum to 1

    w_bin_edges = np.array([120, 200, 300, 400, 500, 600, 700, 800, 900, 920])

    w_bin_centers = (w_bin_edges[:-1] + w_bin_edges[1:]) / 2

    synthetic_w_values = rng.choice(w_bin_centers, size=n_ensemble, p=w_values)

    w_shape, w_loc, w_scale = skewnorm.fit(synthetic_w_values)

    sampled_w = skewnorm.rvs(w_shape, loc=w_loc, scale=w_scale, size=n_ensemble)

    # remove all values below 120 and above 920
    clipped_sampled_w = sampled_w[(sampled_w >= 120) & (sampled_w <= 920)]

    # check if clipped_sampled_w contains enough values, if not, resample and add more values
    while len(clipped_sampled_w) < n_ensemble:
        additional_sample = skewnorm.rvs(w_shape, loc=w_loc, scale=w_scale, size=n_ensemble)
        additional_clipped_sample = additional_sample[(additional_sample >= 120) & (additional_sample <= 920)]
        clipped_sampled_w = np.concatenate([clipped_sampled_w, additional_clipped_sample])
        clipped_sampled_w = clipped_sampled_w[:n_ensemble]

    gamma = clipped_sampled_gamma
    ambient_wind = clipped_sampled_w

    # gamma = np.abs(rng.normal(gamma_init,0.025,n_ensemble))
    # ambient_wind = rng.normal(ambient_wind_init,50,n_ensemble)
    # speed_ensemble = rng.normal(speed,50,n_ensemble)
    
    timesteps = np.arange(kindays_in_min)*res_in_min*60
    timesteps = np.vstack([timesteps]*n_ensemble)
    timesteps = np.transpose(timesteps)

    # accsign = np.ones(n_ensemble)
    # accsign[speed_ensemble < ambient_wind] = -1.

    # compute kinematics

    gamma = gamma * 1e-7

    dv = speed_ensemble - ambient_wind # difference between CME speed and solar wind speed for each ensemble member

    bg_sgn = np.where(speed_ensemble > ambient_wind, 1, -1) # sign of the background speed, 1 if CME is faster than solar wind, -1 otherwise

    cme_r_ensemble = (bg_sgn / gamma) * np.log1p(bg_sgn * gamma * dv * timesteps) + ambient_wind * timesteps + distance0 # (timestamps, ensemblemembers)
    cme_v_ensemble = dv / (1 + bg_sgn * gamma * dv * timesteps) + ambient_wind # (timestamps, ensemblemembers)


    cme_r_mean = np.mean(cme_r_ensemble, axis=1) # (timestamps,) 
    cme_r_std = np.std(cme_r_ensemble, axis=1) # (timestamps,)
    cme_v_mean = np.mean(cme_v_ensemble, axis=1) # (timestamps,)
    cme_v_std = np.std(cme_v_ensemble, axis=1) # (timestamps,)

    cme_r[:,0]= cme_r_mean*u.km.to(u.au) # mean of cme distance ensemble
    cme_r[:,1]=(cme_r_mean - 2*cme_r_std)*u.km.to(u.au)  # lower limit of cme distance ensemble
    cme_r[:,2]=(cme_r_mean + 2*cme_r_std)*u.km.to(u.au) # upper limit of cme distance ensemble
    cme_v[:,0]= cme_v_mean
    cme_v[:,1]=(cme_v_mean - 2*cme_v_std)
    cme_v[:,2]=(cme_v_mean + 2*cme_v_std)
    
    distance_earth, cme_a, cme_b, cme_c = elevo_analytic(cme_r, f, halfwidth, cme_delta)
    distance_earth[cme_hit.all() != 1] = np.nan

    #find next full hour after t0
    format_str = '%Y-%m-%d %H'  
    t0r = datetime.strptime(datetime.strftime(t0, format_str), format_str) + timedelta(hours=1)
    time2=[]
    tstart2=t0r
    tend2=tstart2+timedelta(days=kindays)
    #make 30 min datetimes 
    while tstart2 < tend2:
        time2.append(tstart2)  
        tstart2 += timedelta(minutes=res_in_min)  

    time2_num=parse_time(time2).plot_date        
    time1_num=parse_time(time1).plot_date
    
    if not np.isnan(distance_earth).all():
        results_earth = compute_arrival(distance_earth, cme_v, time1, earth_r[earth_ind])
        results_earth['arr_id'] = [cme_id[0].decode("utf-8")]

        arrival_for_txt = []

        arrival_for_txt.append([
            results_earth['arr_id'][0],
            t0.strftime('%Y-%m-%dT%H:%MZ'),
            "{:.1f}".format(cme_lon[0]),
            "{:.1f}".format(cme_lat[0]),
            "{:.1f}".format(halfAngleCorrected),
            "{:.1f}".format(speed),
            results_earth['arr_time_fin'][0].strftime('%Y-%m-%dT%H:%MZ'),
            "{:.2f}".format(results_earth["arr_time_err_list"][0]),
            "{:.2f}".format(results_earth["arr_speed_list"][0]),
            "{:.2f}".format(results_earth["arr_speed_err_list"][0])
        ])

        df_combined = {
                        "cmeID_elevo": associatedActivityID,
                        "featureCode": featureCode,
                        "L1_ist_obs": earthArrivalTime,
                        "f": f,
                        "halfwidth [°]": np.rad2deg(halfwidth),
                        "longitude [°]": longitudeCorrected,
                        "latitude [°]": latitude,
                        "rinit [km]": distance0,
                        "vinit [km/s]": speed,
                        "tinit [UT]": t0,
                        "gamma [e-7/km]": gamma_init,
                        "solar wind speed [km/s]": ambient_wind_init,
                        "arrival_time [UT]": results_earth["arr_time_fin"][0],
                        "arrival_time_min [UT]": results_earth["arr_time_err0"][0],
                        "arrival_time_max [UT]": results_earth["arr_time_err1"][0],
                        "arrival_hit": results_earth["arr_hit"][0],
                        "arrival_speed [km/s]": results_earth["arr_speed_list"][0],
                        "arrival_speed_err [km/s]": results_earth["arr_speed_err_list"][0],
                        "tnum": np.arange(kindays_in_min) * res_in_min * 60,
                        "r_at_apex [km]": cme_r*AU_in_km,
                        "r_at_target_longitude [km]": distance_earth*AU_in_km,
                        "speed [km/s]": cme_v,
                        }

        time_axis = []
        for t in df_combined['tnum']:
            time_axis.append(df_combined['tinit [UT]'] + timedelta(seconds=int(t)))
        
        df_combined['time_axis [UT]'] = np.array(time_axis)

        sta_lon = get_horizons_coord('STEREO-A',t0).lon.deg

        if sta_lon > 0:
            df_combined['phi [°]'] = sta_lon - df_combined['longitude [°]']
        else:
            df_combined['phi [°]'] = df_combined['longitude [°]'] - sta_lon

        df_combined = get_elevo_t_e_track(df_combined)

    else:
        print(f'CME {associatedActivityID} does not hit (unexplained)')
        results_earth = {
            "arr_time_fin": [np.nan],
            "arr_time_err0": [np.nan],
            "arr_time_err1": [np.nan],
            "arr_time_err_list": [np.nan],
            "arr_id": [np.nan],
            "arr_hit": [np.nan],
            "arr_speed_list": [np.nan],
            "arr_speed_err_list": [np.nan],
        }

        arrival_for_txt = []

        df_combined = {}

    #linear interpolation to time_mat times    
    cme_r = [np.interp(time2_num, time1_num,cme_r[:,i]) for i in range(3)]
    cme_v = [np.interp(time2_num, time1_num,cme_v[:,i]) for i in range(3)]
    cme_lat = np.interp(time2_num, time1_num,cme_lat)
    cme_lon = np.interp(time2_num, time1_num,cme_lon)
    cme_a = [np.interp(time2_num, time1_num,cme_a[:,i]) for i in range(3)]
    cme_b = [np.interp(time2_num, time1_num,cme_b[:,i]) for i in range(3)]
    cme_c = [np.interp(time2_num, time1_num,cme_c[:,i]) for i in range(3)]
    
    time_columns = ['tinit [UT]', 'arrival_time [UT]', 'arrival_time_min [UT]', 'arrival_time_max [UT]', 'time_axis [UT]']
    # convert time columns to strings of format '%Y-%m-%dT%H:%MZ' for better readability in output
    for col in time_columns:
        if col in df_combined:
            if type(df_combined[col]) is list or type(df_combined[col]) is np.ndarray:
                df_combined[col] = np.array([x.strftime('%Y-%m-%dT%H:%MZ') if pd.notnull(x) else pd.NaT for x in df_combined[col]])
            else:
                df_combined[col] = df_combined[col].strftime('%Y-%m-%dT%H:%MZ')
    
    return df_combined
    #return time2_num, cme_r, cme_lat, cme_lon, cme_a, cme_b, cme_c, cme_id, cme_v, halfwidth,results_earth['arr_time_fin'], results_earth['arr_time_err0'], results_earth['arr_time_err1'], results_earth['arr_id'], results_earth['arr_hit'], results_earth['arr_speed_list'], results_earth['arr_speed_err_list']
    #return df_track, df_plot_params

if __name__ == '__main__':
    print("main function here")