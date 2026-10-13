import os
import json
import itertools
import time
import datetime
import requests
import pandas as pd
import numpy as np
import sys
from sunpy.coordinates.ephemeris import get_horizons_coord
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy import units as u
from astropy import constants as const
from sunpy.coordinates import Helioprojective, HeliographicStonyhurst, HeliocentricEarthEcliptic
import pyvista as pv
from matplotlib.path import Path
from scipy.optimize import linear_sum_assignment
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst

RSUN_KM = 695700.0

# HI-1A nominal geometry, from get_hi_params in HI_Data_Processing/functions.py: boresight
# at 14.0037 deg elongation with a 20.275 deg square field, centred on the ecliptic
HI1A_BORESIGHT_ELON = 14.0037
HI1A_FOV_HALFWIDTH = 20.275 / 2
HI1A_ELON_MIN = HI1A_BORESIGHT_ELON - HI1A_FOV_HALFWIDTH
HI1A_ELON_MAX = HI1A_BORESIGHT_ELON + HI1A_FOV_HALFWIDTH

geosphere_purple = '#701F80'
geosphere_green = '#75CC41'
geosphere_yellow = '#F9F200'
geosphere_orange = '#F29707'
my_pink = '#FF69B4'

def download_donki_data(file_path, data_type, start_date='2000-01-01', chunk_days=59, max_retries=3, request_delay=0.5):
    """
    Download DONKI data for the specified data type and save it as a JSON file.

    The DONKI API caps any single request's date range at 60 days, so the full
    range is split into consecutive chunks that are requested separately and
    merged into one JSON array before being written to disk.

    Parameters:
    - file_path: str, the path where the JSON file will be saved.
    - data_type: str, the type of DONKI data to download (e.g., 'CME', 'IPS', etc.).
    - start_date: str, the first date (YYYY-MM-DD) to request data from.
    - chunk_days: int, number of days per request (inclusive span), must be <= 59 to stay under the API's 60-day limit.
    - max_retries: int, number of attempts per chunk before giving up.
    - request_delay: float, seconds to wait between chunk requests.

    """

    data_dir = os.path.dirname(file_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    end_date = datetime.datetime.now()
    chunk_start = datetime.datetime.strptime(start_date, "%Y-%m-%d")

    all_records = []

    while chunk_start <= end_date:
        chunk_end = min(chunk_start + datetime.timedelta(days=chunk_days), end_date)
        url_donki = ('https://ccmc.gsfc.nasa.gov/DONKI-API/get/'+data_type
                     +'?startDate='+chunk_start.strftime("%Y-%m-%d")
                     +'&endDate='+chunk_end.strftime("%Y-%m-%d"))

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url_donki, timeout=600)
                response.raise_for_status()
                chunk_records = response.json()
                break
            except requests.RequestException as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to download DONKI data for {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}: {e}")
                time.sleep(2)

        if chunk_records:
            all_records.extend(chunk_records)

        chunk_start = chunk_end + datetime.timedelta(days=1)
        time.sleep(request_delay)

    # Save merged records as a single JSON file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(all_records, f)

def join_donki_cme_ips_data(donki_df_cmes, donki_df_ips):
    """
    """

    for index, row in donki_df_cmes.iterrows():

        has_earth_arrival = False
        if row['linkedEvents'] is not None:
            linked_events = row['linkedEvents'].copy()
            for num_event, linked_event in enumerate(linked_events):
                if 'IPS' in linked_event['activityID']:
                    matching_ips = donki_df_ips[donki_df_ips['activityID'] == linked_event['activityID']]
                    if not matching_ips.empty:
                        linked_events[num_event] = matching_ips.to_dict('records')[0]
                        if matching_ips.to_dict('records')[0]['location'] == 'Earth':
                            has_earth_arrival = True
                            earth_arrival_time = matching_ips.to_dict('records')[0]['eventTime']
                    
            donki_df_cmes.at[index,'linkedEvents'] = linked_events
        
        donki_df_cmes.at[index,'hasEarthArrival'] = has_earth_arrival
        donki_df_cmes.at[index,'earthArrivalTime'] = earth_arrival_time if has_earth_arrival else None

        cme_analyses = [analysis for analysis in row['cmeAnalyses'] if analysis['isMostAccurate']]

        for analysis in cme_analyses:
            if analysis["time21_5"] is not None and analysis["time21_5"].startswith("00"):
                analysis["time21_5"] = analysis["time21_5"].str.replace("^00", "20", regex=True)

            analysis['associatedActivityID'] = row['activityID']
            # the CME-level start time, i.e. when the CME was first seen in coronagraph
            # imagery; carried onto the analysis so it survives get_cme_analyses_df's explode
            analysis['associatedCMEstartTime'] = row['startTime']

            analysis['arrivalEllipsePossible'] = False
            if (analysis['longitude']) is not None and analysis['halfAngle'] is not None:
                if np.abs(analysis['longitude']) < analysis['halfAngle']:
                    analysis['arrivalEllipsePossible'] = True

        donki_df_cmes.at[index,'cmeAnalyses'] = cme_analyses

    return donki_df_cmes

def get_cme_analyses_df(df_joint):

    all_analyses = []
    for index, row in df_joint.iterrows():
        for analysis in row['cmeAnalyses']:
            analysis['hasEarthArrival'] = row['hasEarthArrival']
            analysis['earthArrivalTime'] = row['earthArrivalTime']
            all_analyses.append(analysis)

    analyses_df = pd.DataFrame(all_analyses)

    
    # the shock is kept where one was analysed, falling back to the leading edge otherwise
    feature_code_priority = {'SH': 0, 'LE': 1}
    analyses_df['_feature_code_priority'] = analyses_df['featureCode'].map(feature_code_priority)

    # keeps only SH and LE features
    analyses_df = analyses_df.dropna(subset=['_feature_code_priority'])
    analyses_df = analyses_df.sort_values('_feature_code_priority')

    # keeps only the first occurrence of each associatedActivityID (the one with the highest priority featureCode)
    analyses_df = analyses_df.drop_duplicates(subset='associatedActivityID', keep='first')
    analyses_df = analyses_df.drop(columns='_feature_code_priority').reset_index(drop=True)

    return analyses_df

def convert_strudl_output(path_to_strudl_dict=None, pa_array=None, use_earth_pa=True,
                          use_max_elon_pa=False, use_central_pa=False, sc='A',
                          strudl_dict=None):
    """
    Converts the output from a STRUDL dictionary file into a track dictionary, optionally using a provided
    position angle (PA) array or calculating the Earth's position angle for each CME.
    Parameters
    ----------
    path_to_strudl_dict : str
        Path to the .npy file containing the STRUDL dictionary with CME data.
    pa_array : array-like, optional
        Array of position angles (in degrees) to use for each CME. If provided, must match the number of CMEs
        in the STRUDL dictionary. If None, Earth's position angle will be used if `use_earth_pa` is True.
    use_earth_pa : bool, default=True
        If True, calculates the Earth's position angle for each CME start time using `get_earth_ecliptic`.
        If False and `pa_array` is None, raises an error.
    use_max_elon_pa : bool, default=False
        If True, uses the position angle corresponding to the largest elongation for each CME.
        Cannot be used simultaneously with `pa_array` or `use_earth_pa`.
    use_central_pa : bool, default=False
        If True, uses the midpoint of the front's overall position angle extent for each CME,
        which is the way HIGeoCAT picks its single J-plot position angle: as near as practicable
        to the CME apex, rather than to the ecliptic. Needs no ephemeris, unlike `use_earth_pa`.
        Cannot be used simultaneously with `pa_array`, `use_earth_pa` or `use_max_elon_pa`.
    sc : str, default='A'
        Spacecraft identifier to use when calculating Earth's position angle.
    strudl_dict : dict, optional
        An already-loaded STRUDL dictionary, used in place of reading `path_to_strudl_dict`.
        The segmentation file is ~1 GB and is cut at more than one position angle, so callers
        that need several cuts should load it once and pass it here.
    Returns
    -------

    ------
    SystemExit
        If more than one of `pa_array` / `use_earth_pa` / `use_max_elon_pa` / `use_central_pa` is
        set, if the length of `pa_array` does not match the number of CMEs, or if none of them
        are set.
    """

    if strudl_dict is None:
        strudl_dict = np.load(path_to_strudl_dict, allow_pickle=True).item()

    if pa_array is not None and (use_earth_pa or use_max_elon_pa or use_central_pa):
        sys.exit("Error: Both pa_array and use_earth_pa/use_max_elon_pa/use_central_pa are set.")

    elif sum([bool(use_earth_pa), bool(use_max_elon_pa), bool(use_central_pa)]) > 1:
        sys.exit("Error: More than one of use_earth_pa/use_max_elon_pa/use_central_pa is set.")

    elif pa_array is not None and len(pa_array) != len(strudl_dict):
        sys.exit("Error: Length of pa_array does not match number of CMEs in STRUDL dictionary.")

    elif pa_array is None and not (use_earth_pa or use_max_elon_pa or use_central_pa):
        sys.exit("Error: Neither pa_array nor use_earth_pa/use_max_elon_pa/use_central_pa are set.")

    if use_earth_pa:
        # get Earth position angle for every time of CME start
        pa_array = []

        for cme_key in strudl_dict.keys():
            cme_start_time = strudl_dict[cme_key]['times'][0]
            earth_pa = get_earth_ecliptic(cme_start_time, sc)
            pa_array.append(earth_pa)

    elif use_max_elon_pa:
        # get position angle of largest elongation for every CME
        pa_array = []

        for cme_key in strudl_dict.keys():
            max_elon = -np.inf
            best_pa = None

            for t_ind in range(len(strudl_dict[cme_key]['times'])):
                pa_temp = strudl_dict[cme_key]['cme_front_wcs'][t_ind][0]
                elon_temp = strudl_dict[cme_key]['cme_front_wcs'][t_ind][1]

                if np.nanmax(elon_temp) > max_elon:
                    max_elon = np.nanmax(elon_temp)
                    best_pa = pa_temp[np.nanargmax(elon_temp)]
            
            pa_array.append(best_pa)

    elif use_central_pa:
        # get the midpoint of the front's position angle extent for every CME, taken over the
        # whole track rather than per time step so the cut sits at one fixed position angle the
        # way a HIGeoCAT J-plot does
        pa_array = []

        for cme_key in strudl_dict.keys():
            pa_min = np.inf
            pa_max = -np.inf

            for t_ind in range(len(strudl_dict[cme_key]['times'])):
                pa_temp = strudl_dict[cme_key]['cme_front_wcs'][t_ind][0]
                pa_min = min(pa_min, np.nanmin(pa_temp))
                pa_max = max(pa_max, np.nanmax(pa_temp))

            pa_array.append((pa_min + pa_max) / 2)

    pa_array = np.array(pa_array)
    
    track_dictionary = {}

    # loop over every CME in STRUDL dictionary and cut along desired position angle
    for cme_ind, cme_key in enumerate(strudl_dict.keys()):
        earth_pa = pa_array[cme_ind]
        elon_temp = []
        time_temp = []
        
        for t_ind in range(len(strudl_dict[cme_key]['times'])):
            pa_temp = strudl_dict[cme_key]['cme_front_wcs'][t_ind][0]
            pa_inds = np.where(np.abs(pa_temp - earth_pa) <= 2.)[0]

            if len(pa_inds) > 0:
                elon_temp.append(np.nanmean(strudl_dict[cme_key]['cme_front_wcs'][t_ind][1][pa_inds]))
                time_temp.append(strudl_dict[cme_key]['times'][t_ind])

                if cme_key not in track_dictionary.keys():
                    track_dictionary[cme_key] = []

        if len(time_temp) > 0:
            track_dictionary[cme_key] = {
            'track_no': 'strudl',
            'time': time_temp,
            'elongation': elon_temp,
            'pa': [earth_pa]*len(time_temp),
            'sc': [sc]*len(time_temp)
            }

    return track_dictionary

def get_stereo_heeq_lon(pos_time, sc, return_radius=False):
    """
    Get the Heliocentric Earth Equatorial (HEEQ) longitude of a STEREO spacecraft at a given time.
    Parameters
    ----------
    pos_time : astropy.time.Time or str
        The time at which to obtain the STEREO spacecraft position. Can be an astropy Time object or a string parseable by astropy.
    sc : str
        The STEREO spacecraft identifier. Must be either 'A' for STEREO-A or 'B' for STEREO-B.
    return_radius : bool, default=False
        If True, also return the spacecraft's heliocentric distance in AU. The batched Horizons
        query already fetches the full coordinate, so this costs no extra request.
    Returns
    -------
    float
        The longitude of the STEREO spacecraft in Heliocentric Earth Equatorial (HEEQ) coordinates, in degrees.
        If `return_radius` is True, a (longitude, radius) tuple of arrays instead.
    Notes
    -----
    This function requires the `get_horizons_coord` function to retrieve the spacecraft position and assumes
    that `SkyCoord`, `HeliographicStonyhurst`, and `u` (astropy units) are available.
    """

    try:
        iter(pos_time)
    except TypeError:
        pos_time = [pos_time]

    # get STEREO position at given time
    if sc == 'A': 
        stereo_pos = get_horizons_coord('STEREO-A', pos_time)

    if sc == 'B':
        stereo_pos = get_horizons_coord('STEREO-B', pos_time)

    # transform STEREO position to Heliocentric Earth Equatorial coordinates
    # get longitude in degrees
    stereo_heeq = [SkyCoord(pos).transform_to(HeliographicStonyhurst()) for pos in stereo_pos]
    stereo_heeq_lon = np.array([pos.lon.to(u.deg).value for pos in stereo_heeq])

    if return_radius:
        return stereo_heeq_lon, np.array([pos.radius.to(u.AU).value for pos in stereo_heeq])

    return stereo_heeq_lon

def make_cme_lemniscate(edge_distance, angular_width, lon, lat,resolution=200):

    w_distance = edge_distance * np.tan(np.deg2rad(angular_width / 2))

    # Lemniscate definition
    theta = np.linspace(0, np.pi/2, resolution)
    phi   = np.linspace(0, 2*np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)

    x = edge_distance * np.cos(theta)
    y = w_distance * np.cos(theta) * np.sin(theta) * np.cos(phi)
    z = w_distance * np.cos(theta) * np.sin(theta) * np.sin(phi)

    pts = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    
    Rmat = rot_z(lon) @ rot_y(-lat) 
    pts_rot = Rmat @ pts

    xr = pts_rot[0].reshape(x.shape)
    yr = pts_rot[1].reshape(y.shape)
    zr = pts_rot[2].reshape(z.shape)

    return xr, yr, zr

# Rotations
def rot_z(lon):
    lon_rad = np.deg2rad(lon)
    return np.array([
        [ np.cos(lon_rad), -np.sin(lon_rad), 0],
        [ np.sin(lon_rad),  np.cos(lon_rad), 0],
        [ 0,            0,           1]
    ])

def rot_y(lat):
    lat_rad = np.deg2rad(lat)
    return np.array([
        [ np.cos(lat_rad), 0, np.sin(lat_rad)],
        [ 0,           1, 0],
        [-np.sin(lat_rad), 0, np.cos(lat_rad)]
    ])


def sphere_to_cart_heeq(r, lat_rad, lon_rad):
    """Convert spherical coordinates (r, lat, lon) to Cartesian coordinates (x, y, z) in HEEQ system."""

    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return x, y, z

def cart_heeq_to_sphere(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y,x)
    phi = np.arctan2(z,np.sqrt(x**2 + y**2))
    return r,phi,theta

def ecliptic_normal_heeq(obstime):
    """Unit normal of the ecliptic plane (ecliptic north) in HEEQ Cartesian coordinates.
    """
    pole = SkyCoord(lon=0*u.deg, lat=90*u.deg, distance=1*u.AU,
                    frame=HeliocentricEarthEcliptic(obstime=obstime))
    n = pole.transform_to(HeliographicStonyhurst(obstime=obstime)).cartesian.xyz.value
    return n / np.linalg.norm(n)

def make_circle(radius,center_x, center_y,num_points):
    theta = np.linspace(0, 2 * np.pi, num_points)

    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    return x, y

def make_cme_ellipse(edge_distance, half_width_deg, lon, lat, aspect_ratio_inv=0.7, resolution=100):
    """
    3D CME ellipsoid using ElEvo geometry (Möstl et al. 2015, eqs. 3, 6, 7).

    edge_distance    : apex distance from Sun (Rsun) = R(t)
    half_width_deg   : half angular width lambda (degrees)
    lon, lat         : propagation direction in HEEQ (degrees)
    aspect_ratio_inv : f = b/a, radial-to-lateral semi-axis ratio (default 0.7)
    """
    R   = float(edge_distance)
    f   = float(aspect_ratio_inv)
    lam = np.deg2rad(float(half_width_deg))

    # ElEvo equations
    theta_T = np.arctan(f**2 * np.tan(lam))     
    omega   = np.sqrt((f**2 - 1) * np.cos(theta_T)**2 + 1)               
    b_axis  = R * omega * np.sin(lam) / (np.cos(lam - theta_T) + omega * np.sin(lam))  
    a_axis  = b_axis / f            
    c_dist  = R - b_axis      

    theta = np.linspace(0, np.pi, resolution)
    phi   = np.linspace(0, 2 * np.pi, resolution)
    theta, phi = np.meshgrid(theta, phi)

    x = c_dist + b_axis * np.cos(theta)
    y = a_axis * np.sin(theta) * np.cos(phi) 
    z = f * a_axis * np.sin(theta) * np.sin(phi)

    pts     = np.vstack([x.ravel(), y.ravel(), z.ravel()])
    Rmat    = rot_z(lon)@rot_y(-lat) 

    pts_rot = Rmat @ pts

    xr = pts_rot[0].reshape(x.shape)
    yr = pts_rot[1].reshape(y.shape)
    zr = pts_rot[2].reshape(z.shape)

    return xr, yr, zr

def get_earth_ecliptic(pos_time, sc):
    """
    Calculates the position angle of Earth in ecliptic coordinates as seen from a specified STEREO spacecraft at a given time.
    Parameters
    ----------
    pos_time : datetime.datetime
        The time at which to compute the Earth's position.
    sc : str
        The STEREO spacecraft identifier. Must be either 'A' (for STEREO-A) or 'B' (for STEREO-B).
    Returns
    -------
    earth_pa : float
        The position angle of Earth in degrees, as seen from the specified STEREO spacecraft.
    Notes
    -----
    Requires the functions `get_body_heliographic_stonyhurst` and `get_horizons_coord` to obtain the positions of Earth and STEREO spacecraft, respectively.
    """

    # get Earth position at given time
    earth_pos = get_body_heliographic_stonyhurst('earth', pos_time)

    # get STEREO position at given time
    if sc == 'A': 
        stereo_pos = get_horizons_coord('STEREO-A', pos_time)

    if sc == 'B':
        stereo_pos = get_horizons_coord('STEREO-B', pos_time)

    # transform Earth position to Helioprojective coordinates as seen from STEREO
    earth_hpc = SkyCoord(earth_pos).transform_to(Helioprojective(observer=stereo_pos))

    earth_x = earth_hpc.Tx.to(u.deg).value*np.pi/180
    earth_y = earth_hpc.Ty.to(u.deg).value*np.pi/180

    # convert to position angle
    earth_pa = np.rad2deg(np.arctan2(-np.cos(earth_y)*np.sin(earth_x), np.sin(earth_y)))
    
    return earth_pa

def calculate_corrected_initial_distance(edge_distance, angular_width, lon, lat, time_str, earth_coord, make_plot=False,stereo_coords=None):

    color_ellipse = geosphere_green
    color_lemniscate = my_pink
    color_intersection_ecliptic = my_pink
    color_circles='dodgerblue'

    plot_stereo = False

    if make_plot:
        pv.global_theme.multi_rendering_splitting_position = 0.40
        plotter = pv.Plotter(off_screen=False,line_smoothing=True,image_scale=10,shape='2|1')
        plotter.window_size = [2400, 1600]

        plotter.background_color = "white"


    AU_in_Rsun = (1 * u.AU).to(u.solRad).value
    
    # CME parameters
    x_cme, y_cme, z_cme = make_cme_lemniscate(edge_distance, angular_width, lon, lat,100)
    lemniscate_mesh = pv.StructuredGrid(x_cme, y_cme, z_cme)

    # Sun
    r_sun = 1
    Sun = pv.Sphere(radius=r_sun)

    # Earth
    ex, ey, ez = [earth_coord.radius.value*AU_in_Rsun, np.deg2rad(earth_coord.lat.value), np.deg2rad(earth_coord.lon.value)]
    ecliptic_normal = ecliptic_normal_heeq(earth_coord.obstime)
    ex_cart, ey_cart, ez_cart = sphere_to_cart_heeq(ex, ey, ez)

    Earth = pv.Sphere(radius=1.0*r_sun,center=(ex_cart, ey_cart, ez_cart))

    # STEREO-A
    if plot_stereo:

        stereo_x, stereo_y, stereo_z = [stereo_coords.radius.value*AU_in_Rsun, np.deg2rad(stereo_coords.lat.value), np.deg2rad(stereo_coords.lon.value)]
        stereo_x_cart, stereo_y_cart, stereo_z_cart = sphere_to_cart_heeq(stereo_x, stereo_y, stereo_z)
        sta = pv.Sphere(radius=1.0*r_sun,center=(stereo_x_cart, stereo_y_cart, stereo_z_cart))
    
    x = np.linspace(-1.0*AU_in_Rsun,AU_in_Rsun, 80)
    y = np.linspace(-1.0*AU_in_Rsun,AU_in_Rsun, 80)

    # Ecliptic
    x_ecliptic, y_ecliptic = np.meshgrid(x, y)
    z_ecliptic = - ( ecliptic_normal[0] * x_ecliptic +  ecliptic_normal[1] * y_ecliptic) / ecliptic_normal[2]
    ecliptic = pv.StructuredGrid(x_ecliptic, y_ecliptic, z_ecliptic)

    try:
        # this intersects two meshes and returns a line of points at their intersection (based on resolution of lemniscale)
        intersection, s1_split, s2_split = lemniscate_mesh.extract_surface().triangulate().intersection(ecliptic.extract_surface().triangulate(),False,False,True)
        pts_intersection = intersection.points

    except:
        print("failed first intersection")
        exit()


    #computing radial distances of all these intersection points
    radial_distance_intersection,lat_intersection,lon_intersection = cart_heeq_to_sphere(pts_intersection[:,0], pts_intersection[:,1], pts_intersection[:,2])
    lat_intersection = np.rad2deg(lat_intersection)
    lon_intersection = np.rad2deg(lon_intersection)

    radial_distance_cme, lat_cme,lon_cme = cart_heeq_to_sphere(x_cme, y_cme, z_cme)
    lat_cme = np.rad2deg(lat_cme)
    lon_cme = np.rad2deg(lon_cme)

    max_angle = angular_width
    apex_long = lon

    if len(radial_distance_intersection)>0:

        min_radii = 3.0
        pts_farenough = pts_intersection[radial_distance_intersection>min_radii]
        lon_farenough = lon_intersection[radial_distance_intersection>min_radii]
        lat_farenough = lat_intersection[radial_distance_intersection>min_radii]
        radial_enough = radial_distance_intersection[radial_distance_intersection>min_radii]
       
        # apex_lemniscate = sphere_to_cart_heeq(edge_distance, np.deg2rad(lat),np.deg2rad(lon))

        if lon_farenough.shape[0]>0:

            ## FIRST we calculate the angular width of the intersection to ecliptic, and we take the middle point as the APEX
            # The intersection longitudes come from arctan2, so a CME propagating near +-180
            # so min/max would span ~360 deg. Must be unwrapped to get the correct angular width and apex longitude.
            
            delta_lon_farenough = (lon_farenough - lon + 180.0) % 360.0 - 180.0
            min_lon_intersection  = lon + np.min(delta_lon_farenough)
            max_lon_intersection  = lon + np.max(delta_lon_farenough)

            max_angle = (max_lon_intersection-min_lon_intersection)/2.0
            apex_long = (min_lon_intersection + max_angle + 180.0) % 360.0 - 180.0

            delta_lon_all = (lon_intersection - lon + 180.0) % 360.0 - 180.0
            idx_closest_to_mean = np.argmin(np.abs(delta_lon_all))
            radial_closest_to_mean = radial_distance_intersection[idx_closest_to_mean]
            lat_apex = lat_intersection[idx_closest_to_mean]

            ## this is the new APEX of the CME  allegedly  (mid point between extremas of intersection longitude)
            x_a,y_a,z_a = sphere_to_cart_heeq(radial_closest_to_mean,np.deg2rad(lat_apex),np.deg2rad(apex_long))
            apex_pts2 = np.array([x_a,y_a,z_a])

            ### HAVE TO CHOOSE HERE EITHER ANGLE OF THE INTERSECTION WITH ECLIPTIC OR WE TAKE THE ANGLE AT THE LARGEST PART OF THE INTERSECTION LIKE JURGEN DID IN HIS PAPER 
            distances_to_mid_line = np.cross(pts_intersection, apex_pts2)
            distances = np.linalg.norm(distances_to_mid_line, axis=1) / np.linalg.norm(apex_pts2)
            idx_max_distance = np.argmax(distances)

            new_apex2 = pv.Sphere(radius=0.6*r_sun,center=(x_a,y_a,z_a))
            x1,y1,_ =sphere_to_cart_heeq(radial_closest_to_mean,np.deg2rad(lat_apex),np.deg2rad(min_lon_intersection))
            z1 = - ( ecliptic_normal[0] * x1 +  ecliptic_normal[1] * y1) / ecliptic_normal[2]

            arrow_limit_cme_1 = pv.Arrow(
                start=(0.0,0.0,0.0), 
                direction=(x1,y1,z1) , 
                scale=21.5,
                tip_length=0.001,    # Proportion of arrow that is the head
                tip_radius=0.001,   # Radius of the arrow head
                shaft_radius=0.01  # Radius of the arrow shaft
            )

            x2,y2,_ =sphere_to_cart_heeq(radial_closest_to_mean,np.deg2rad(lat_apex),np.deg2rad(max_lon_intersection))
            z2 = - (ecliptic_normal[0] * x2 +  ecliptic_normal[1] * y2) / ecliptic_normal[2]

            arrow_limit_cme_2 = pv.Arrow(
                start=(0.0,0.0,0.0), 
                direction=(x2,y2,z2), 
                scale=21.5,
                tip_length=0.001,    
                tip_radius=0.001,  
                shaft_radius=0.01 
            )

            x_ell, y_ell, z_ell = make_cme_ellipse(radial_closest_to_mean, max_angle, apex_long, earth_coord.lat.value, resolution=50)#
            ellipse = pv.StructuredGrid( x_ell, y_ell, z_ell)

            try:
                intersection2, s1_split, s2_split = ellipse.extract_surface().triangulate().intersection(ecliptic.extract_surface().triangulate(),False,False,True)
            except:
                print("failed second intersection")


        if make_plot:
            ## We compute the first view point as 90. degrees from propagation direction and ON the ecliptic
            first_angle = lon+90.0

            numerator = -(ecliptic_normal[0] * np.cos(np.deg2rad(earth_coord.lon.value)) + ecliptic_normal[1] * np.sin(np.deg2rad(earth_coord.lon.value)))
            # Compute the latitude
            latitude_rad = np.arctan2(numerator, ecliptic_normal[2])

            first_x,first_y,_ = sphere_to_cart_heeq(70.0, latitude_rad, np.deg2rad(first_angle))
            first_z = - ( ecliptic_normal[0] * first_x +  ecliptic_normal[1] * first_y) / ecliptic_normal[2]

            second_x,second_y,second_z = sphere_to_cart_heeq(100.0, np.deg2rad(45.0), np.deg2rad(apex_long))
            second_z = - ( ecliptic_normal[0] * second_x +  ecliptic_normal[1] * second_y) / ecliptic_normal[2]

            third_x,third_y,third_z = sphere_to_cart_heeq(120.0, np.deg2rad(90.0), np.deg2rad(earth_coord.lon.value))

            # ex_cart, ey_cart, ez_cart
            camera_positions = [ [(first_x,first_y,first_z),(13.0,0.0,0.0),(0.0,0.0,1.0)], 
                                [(second_x,second_y,second_z),(0.0,0.0,0.0),(0.0,0.0,1.0)], 
                                [(third_x,third_y,third_z),(0.0,0.0,0.0),(-1.0,0.0,0.0)]]

            for idx_plot in range(0,3):
                plotter.subplot(idx_plot)
        
                for i in range(1,25):
                    x_circle,y_circle = make_circle(i*10.0,0.0, 0.0,100 )
                    z_circle = - ( ecliptic_normal[0] * x_circle +  ecliptic_normal[1] * y_circle) / ecliptic_normal[2]
                    circle = pv.StructuredGrid(x_circle, y_circle, z_circle )
                    plotter.add_mesh(circle, color=color_circles, show_edges=True)
                    if idx_plot==2 :
                        tip = (-i*10.0 +2.0, 0, 0)
                        label = pv.Label(str(i*10.0)+" Rs", position=tip,size=18)
                        plotter.add_actor(label)

                ## plotting the axis of propagation of DONKI CME 
                x_axis_propag, y_axis_propag, z_axis_propag = sphere_to_cart_heeq(edge_distance, np.deg2rad(lat), np.deg2rad(lon))

                arrow_cme_propagation = pv.Arrow(
                    start=(0.0,0.0,0.0), 
                    direction=(x_axis_propag, y_axis_propag, z_axis_propag), 
                    scale=30.0,
                    tip_length=0.05,  
                    tip_radius=0.03,
                    shaft_radius=0.01 
                )

                plotter.add_mesh(arrow_cme_propagation, color=color_lemniscate, show_edges=False)

                if plot_stereo:
                    arrow_stereo = pv.Arrow(
                        start=(0.0,0.0,0.0), 
                        direction=(stereo_x_cart, stereo_y_cart, stereo_z_cart), 
                        scale=30.0,
                        tip_length=0.05,    
                        tip_radius=0.03,   
                        shaft_radius=0.01 
                    )


                arrow_normal = pv.Arrow(
                    start=(0.0,0.0,0.0), 
                    direction=(0.0,0.0,1.0), 
                    scale=30.0,
                    tip_length=0.05,  
                    tip_radius=0.03,  
                    shaft_radius=0.01  
                )

                plotter.add_mesh(arrow_normal, color=geosphere_yellow, show_edges=False)


                arrow_earth = pv.Arrow(
                    start=(0.0,0.0,0.0), 
                    direction=(ex_cart, ey_cart, ez_cart), 
                    scale=30.0,
                    tip_length=0.05,  
                    tip_radius=0.03, 
                    shaft_radius=0.01 
                )

                if plot_stereo:
                    stereo_pos = np.array([stereo_x_cart,stereo_y_cart,stereo_z_cart])

                    dist_stereo = np.dot(stereo_pos, ecliptic_normal)
                    r_stereo = np.linalg.norm(stereo_pos)
                    lat_stereo = np.rad2deg(np.arcsin(dist_stereo / r_stereo))

                if idx_plot==2 or idx_plot==1:
                    plotter.add_mesh(arrow_earth, color=color_circles, show_edges=False, opacity=0.5)
                    tip = (ex_cart+5.0, ey_cart, ez_cart)
                    label = pv.Label("To Earth", position=tip,size=17)
                    plotter.add_actor(label)

                    if plot_stereo:
                        print(f"Calculated STEREO A HEE Latitude: {lat_stereo:.3f} degrees for ",stereo_x_cart)

                        plotter.add_mesh(arrow_stereo, color='red', show_edges=False)
                        tip = (stereo_x_cart+5.0, stereo_y_cart, stereo_z_cart)
                        label = pv.Label("To StereoA", position=tip,size=17)
                        plotter.add_actor(label)


                plotter.add_mesh(Sun, color=geosphere_yellow, show_edges=False)
                plotter.add_mesh(Earth, color=color_circles, show_edges=False)
                plotter.add_mesh(lemniscate_mesh.extract_surface(), color=color_lemniscate,opacity=0.20, show_edges=True)

                if plot_stereo:
                    plotter.add_mesh(sta, color='red', show_edges=False)

                if len(radial_distance_intersection)>0 and lon_farenough.shape[0]>0 and len(intersection2.points)>0:

                    plotter.add_mesh(intersection, color=color_intersection_ecliptic, line_width=6)
                    plotter.add_mesh(intersection2, color=color_ellipse, line_width=8)
                    plotter.add_mesh(new_apex2, color=color_ellipse, show_edges=False)

                    if (idx_plot==2 or idx_plot==1):
                        plotter.add_mesh(arrow_limit_cme_1, color='black',opacity=1, show_edges=False)
                        plotter.add_mesh(arrow_limit_cme_2, color='black',opacity=1, show_edges=False)


                plotter.camera.position = camera_positions[idx_plot][0]
                plotter.camera.focal_point = camera_positions[idx_plot][1]
                plotter.camera.up = camera_positions[idx_plot][2]            

            if not os.path.exists('images'):
                os.makedirs('images')

            chosen_file = 'img2025-03-13T22:46Z-lon: -4.0(-3.7)-lat:26.0-width:74.0(33.8).pdf'
            given_file_name = 'img'+time_str+'-lon: '+str(lon)+'('+format(float(apex_long),".1f")+')'+'-lat:'+str(lat)+'-width:'+str(angular_width)+'('+format(float(max_angle*2.0),".1f")+').pdf'

            if given_file_name == chosen_file:
                plotter.save_graphic('../paper_plots/img'+time_str+'-lon: '+str(lon)+'('+format(float(apex_long),".1f")+')'+'-lat:'+str(lat)+'-width:'+str(angular_width)+'('+format(float(max_angle*2.0),".1f")+').pdf',raster=False)    # plotter.show()

            plotter.save_graphic('images/'+given_file_name,raster=False)    # plotter.show()
            plotter.close()
            #plotter.show()

    if len(radial_distance_intersection)>0 and lon_farenough.shape[0]>0:
        return radial_closest_to_mean, max_angle, apex_long
    else:
        return None, None, None
    
def project_to_hi1a_frame(xyz_rsun, sta_coord, earth_coord):
    """
    Project 3D HEEQ points into HI-1A's view, in degrees.

    Returns (u, v): u along the ecliptic towards Earth, v perpendicular to it.
    """
    r, lat, lon = cart_heeq_to_sphere(xyz_rsun[0].ravel(), xyz_rsun[1].ravel(), xyz_rsun[2].ravel())

    pts = SkyCoord(np.rad2deg(lon) * u.deg, np.rad2deg(lat) * u.deg, r * u.solRad,
                   frame=HeliographicStonyhurst(obstime=sta_coord.obstime))

    # transform to Helioprojective coordinates as seen from STEREO-A
    hpc = pts.transform_to(Helioprojective(observer=sta_coord, obstime=sta_coord.obstime))
    tx = hpc.Tx.to_value(u.deg)
    ty = hpc.Ty.to_value(u.deg)

    earth_hpc = SkyCoord(earth_coord).transform_to(
        Helioprojective(observer=sta_coord, obstime=sta_coord.obstime))
    psi = np.arctan2(earth_hpc.Ty.to_value(u.deg), earth_hpc.Tx.to_value(u.deg))

    return (tx * np.cos(psi) + ty * np.sin(psi),
            -tx * np.sin(psi) + ty * np.cos(psi))


def _in_hi1a_fov_3d(xyz_rsun, sta_coord, earth_coord, margin_deg=0.0):
    """
    Whether the CME body described by the 3D HEEQ points `xyz_rsun` reaches HI-1A's FOV.

    Returns (mask, u, v) where mask flags the individual points inside the FOV. A CME can
    also be large enough to engulf the FOV without any surface point landing inside it, so
    callers should use `hi1a_fov_reached` rather than mask.any() alone.
    """
    u_ax, v_ax = project_to_hi1a_frame(xyz_rsun, sta_coord, earth_coord)

    mask = ((u_ax >= HI1A_ELON_MIN - margin_deg) & (u_ax <= HI1A_ELON_MAX + margin_deg) &
            (np.abs(v_ax) <= HI1A_FOV_HALFWIDTH + margin_deg))
    return mask, u_ax, v_ax


def hi1a_fov_reached(xyz_rsun, sta_coord, earth_coord, margin_deg=0.0):
    """True if the CME overlaps HI-1A's FOV, either partially or by covering it entirely."""
    mask, u_ax, v_ax = _in_hi1a_fov_3d(xyz_rsun, sta_coord, earth_coord, margin_deg)
    if mask.any():
        return True

    # the projection of a convex body is convex, so a CME larger than the FOV shows up as
    # the FOV corners falling inside the projected outline rather than the other way round
    points = np.column_stack([u_ax, v_ax])
    if len(points) < 3:
        return False
    try:
        hull = ConvexHull(points)
    except Exception:
        return False
    corners = np.array([[HI1A_ELON_MIN, -HI1A_FOV_HALFWIDTH], [HI1A_ELON_MIN, HI1A_FOV_HALFWIDTH],
                        [HI1A_ELON_MAX, HI1A_FOV_HALFWIDTH], [HI1A_ELON_MAX, -HI1A_FOV_HALFWIDTH]])
    return bool(Path(points[hull.vertices]).contains_points(corners).any())


def hi1a_fov_entry_time(cme, time_grid, sta_coords_grid, earth_coords_grid,
                        max_age_days=3.0, f=0.7, margin_deg=0.0, resolution=40):
    """
    First time in `time_grid` within [time21_5, time21_5 + max_age_days] at which the 3D
    ElEvo ellipsoid of `cme` (dict with time21_5, longitude, latitude, halfAngle, speed;
    constant speed from 21.5 Rsun) reaches HI-1A's field of view.
    """
    if any(cme.get(k) is None for k in ("time21_5", "longitude", "latitude", "halfAngle", "speed")):
        return None, None
    if not 0 < cme["halfAngle"] < 90:
        return None, None

    t0 = Time(cme["time21_5"].replace("Z", ""))
    mask = (time_grid >= t0) & (time_grid <= t0 + max_age_days * u.day)
    if not np.any(mask):
        return None, None

    # propagate CME outwards for max_age_days and check if any of the resulting ellipsoids reach HI-1A's FOV
    for t, sta, earth in zip(time_grid[mask], sta_coords_grid[mask], earth_coords_grid[mask]):
        R = 21.5 + cme["speed"] * (t - t0).to_value(u.s) / RSUN_KM
        if R <= 1.0:
            continue

        # make 3D ellipse
        xyz = make_cme_ellipse(R, cme["halfAngle"], cme["longitude"], cme["latitude"],
                               aspect_ratio_inv=f, resolution=resolution)
        if hi1a_fov_reached(np.array(xyz), sta, earth, margin_deg=margin_deg):
            return t, R

    return None, None


def convert_strudl_output_start_time_only(path_to_strudl_dict=None, sc='A', strudl_dict=None):
    """
    Converts the output from a STRUDL dictionary file into a track dictionary, optionally using a provided
    position angle (PA) array or calculating the Earth's position angle for each CME.
    Parameters
    ----------
    path_to_strudl_dict : str
        Path to the .npy file containing the STRUDL dictionary with CME data.
    sc : str, default='A'
        Spacecraft identifier to use when calculating Earth's position angle.
    strudl_dict : dict, optional
        An already-loaded STRUDL dictionary, used in place of reading `path_to_strudl_dict`.

    Outputs
    -------
    Saves a numpy file with the start times of each CME in the STRUDL dictionary.
    """

    # loop over every CME in STRUDL dictionary and cut along desired position angle
    if strudl_dict is None:
        strudl_dict = np.load(path_to_strudl_dict, allow_pickle=True).item()
    strudl_times = [[key, 'strudl', strudl_dict[key]['times'][0], sc] for key in strudl_dict.keys()]
    track_df = pd.DataFrame(strudl_times, columns=['track_key', 'track_no', 'first_hi_time', 'sc'])

    return track_df

def fpf_elongation(seconds_since_start, phi, speed, launch_offset, observer_distance_au):
    """
    Fixed-Phi time-elongation profile: a point moving radially at constant speed in a fixed
    direction (Rouillard et al., 2008).

    `phi` is the angle between the observer-Sun line and the propagation direction in radians,
    `speed` is in AU/s, `launch_offset` and `seconds_since_start` are seconds relative to the
    first track point, and the returned elongation is in radians.
    """

    travel_time = seconds_since_start - launch_offset

    return np.arctan((speed * travel_time * np.sin(phi)) /
                     (observer_distance_au - speed * travel_time * np.cos(phi)))


def fit_fpf_track(track_times, track_elongations, observer_distance_au, min_points=4):
    """
    Unconstrained Fixed-Phi fit of one STRUDL time-elongation track.

    Returns a dict with phi_deg, speed_kms, launch_time, rms_deg, on_bound, n_points and
    converged; everything but n_points and converged is None when the fit does not converge.
    """

    empty = {'phi_deg': None, 'speed_kms': None, 'launch_time': None, 'rms_deg': None,
             'on_bound': None, 'n_points': len(track_times), 'converged': False}

    if len(track_times) < min_points:
        return empty

    times = pd.to_datetime(pd.Series(list(track_times)))
    seconds = (times - times.iloc[0]).dt.total_seconds().to_numpy()
    elongations_rad = np.deg2rad(np.asarray(track_elongations, dtype=float))

    if not np.all(np.isfinite(seconds)) or not np.all(np.isfinite(elongations_rad)):
        return empty

    au_km = const.au.to_value('km')
    initial = [np.deg2rad(60.), 500. / au_km, -20. * 3600.]
    lower = [0., 200. / au_km, -20. * 3600.]
    upper = [np.pi, 4000. / au_km, 0.]

    try:
        params, _ = curve_fit(
            lambda x, phi, speed, launch_offset: fpf_elongation(x, phi, speed, launch_offset,
                                                                observer_distance_au),
            seconds, elongations_rad, p0=initial, bounds=(lower, upper), maxfev=20000)
    except (RuntimeError, ValueError):
        return empty

    phi, speed, launch_offset = params
    residuals = fpf_elongation(seconds, phi, speed, launch_offset, observer_distance_au) - elongations_rad

    # relative tolerance on the speed bounds, absolute on the angles and the launch offset,
    # since those are bounded at values whose scale does not vary
    on_bound = bool(phi < np.deg2rad(0.5) or phi > np.pi - np.deg2rad(0.5)
                    or speed < lower[1] * 1.001 or speed > upper[1] * 0.999
                    or launch_offset < lower[2] + 60. or launch_offset > upper[2] - 60.)

    return {'phi_deg': float(np.rad2deg(phi)),
            'speed_kms': float(speed * au_km),
            'launch_time': times.iloc[0] + pd.Timedelta(seconds=float(launch_offset)),
            'rms_deg': float(np.rad2deg(np.sqrt(np.mean(residuals ** 2)))),
            'on_bound': on_bound,
            'n_points': len(track_times),
            'converged': True}


def add_fpf_fits(df_pred, strudl_track_dictionary, sta_radii_au, identifier_predicted='track_key'):
    """
    Fits every STRUDL track in `strudl_track_dictionary` and writes the results onto `df_pred`.

    `df_pred` already carries one row per track, so putting the fit there lets the matching reuse
    match_cmes unchanged, with fpf_launch_time as just another predicted-time column. Tracks
    missing from the dictionary, or too short to fit, get NaT -- which match_cmes already
    sentinels as an impossible cost.

    `sta_radii_au` maps each track key to the observer's heliocentric distance in AU; see
    get_stereo_heeq_lon(..., return_radius=True).
    """

    fits = []
    for track_key in df_pred[identifier_predicted]:
        track = strudl_track_dictionary.get(track_key)
        if track is None or track_key not in sta_radii_au:
            fits.append({'phi_deg': None, 'speed_kms': None, 'launch_time': None, 'rms_deg': None,
                         'on_bound': None, 'n_points': 0, 'converged': False, 'central_pa': None})
            continue

        fit = fit_fpf_track(track['time'], track['elongation'], sta_radii_au[track_key])
        fit['central_pa'] = track['pa'][0] if len(track['pa']) > 0 else None
        fits.append(fit)

    df_pred['fpf_launch_time'] = pd.to_datetime([fit['launch_time'] for fit in fits])
    df_pred['fpf_phi_deg'] = [fit['phi_deg'] for fit in fits]
    df_pred['fpf_speed_kms'] = [fit['speed_kms'] for fit in fits]
    df_pred['fpf_rms_deg'] = [fit['rms_deg'] for fit in fits]
    df_pred['fpf_on_bound'] = [fit['on_bound'] for fit in fits]
    df_pred['fpf_n_points'] = [fit['n_points'] for fit in fits]
    df_pred['fpf_central_pa'] = [fit['central_pa'] for fit in fits]

    converged = sum(fit['converged'] for fit in fits)
    on_bound = sum(bool(fit['on_bound']) for fit in fits)
    print(f"Fixed-Phi fits: {converged}/{len(fits)} converged, {on_bound} against a parameter bound")

    return df_pred


def match_cmes(predicted_cmes, ground_truth_cmes, match_key_ground_truth='associatedCMEstartTime', match_key_predicted='predicted_launch_time', identifier_ground_truth='associatedCMEID', identifier_predicted='track_key', threshold=60*12):
    """
    Matches predicted CMEs to ground truth CMEs based on temporal proximity.
    This function constructs a cost matrix using the absolute time difference between predicted and ground truth CME times (depending on chosen criterion).
    It then uses the Hungarian algorithm (linear sum assignment) to find the optimal one-to-one matching between predicted and ground truth CMEs,
    subject to a maximum allowed time difference (threshold). Only matches within the threshold are considered valid.
    Parameters
    ----------
    predicted_cmes : pandas.DataFrame
        DataFrame containing predicted CME events. Must include columns specified by `identifier_predicted` and `match_key_predicted`.
    ground_truth_cmes : pandas.DataFrame
        DataFrame containing ground truth CME events. Must include columns specified by `identifier_ground_truth` and `match_key_ground_truth`.
    match_key_ground_truth : str, optional
        Column name in `ground_truth_cmes` for the CME start time (default: 'associatedCMEstartTime').
    match_key_predicted : str, optional
        Column name in `predicted_cmes` for the predicted CME launch time (default: 'predicted_launch_time').
    identifier_ground_truth : str, optional
        Column name in `ground_truth_cmes` for unique CME identifiers (default: 'associatedCMEID').
    identifier_predicted : str, optional
        Column name in `predicted_cmes` for unique CME identifiers (default: 'track_key').
    threshold : int, optional
        Maximum allowed time difference (in minutes) between predicted and ground truth CME start times for a valid match (default: 60*12).
    Returns
    -------
    matches : dict
        Dictionary mapping each matched predicted CME identifier to a dictionary with match details, including ground truth match, times, and time difference.
    matched_pred : set
        Set of predicted CME identifiers that were successfully matched.
    matched_gt : set
        Set of ground truth CME identifiers that were successfully matched.
    Notes
    -----
    - Only one-to-one matches are returned.
    - Matches with time differences greater than `threshold` or with missing times are ignored.

    """

    # Extract only the needed columns as numpy arrays
    pred_ids   = predicted_cmes[identifier_predicted].to_numpy()
    pred_times = predicted_cmes[match_key_predicted].values
    gt_ids     = ground_truth_cmes[identifier_ground_truth].to_numpy()
    gt_times   = ground_truth_cmes[match_key_ground_truth].values

    # Convert to datetime64[ns] if not already
    if not np.issubdtype(pred_times.dtype, np.datetime64):
        pred_times = pd.to_datetime(pred_times)
    if not np.issubdtype(gt_times.dtype, np.datetime64):
        gt_times   = pd.to_datetime(gt_times)

    # Build cost matrix: absolute time diffs in minutes

    diffs = np.abs(pred_times[:, None] - gt_times[None, :])
    cost_matrix = diffs.astype('timedelta64[m]').astype(float)

    # a NaT on either side casts to a large negative sentinel rather than to NaN, so np.isnan
    # does not see it and the threshold test below would read it as the cheapest pairing there
    # is -- i.e. an event with no time would be matched in preference to one that fits
    missing_time = pd.isna(pred_times)[:, None] | pd.isna(gt_times)[None, :]

    # Apply threshold (minutes); set large cost where invalid
    cost_matrix[np.logical_or.reduce([missing_time,
                                      cost_matrix > threshold,
                                      np.isnan(cost_matrix)])] = 1e10

    # ---- Solve assignment ----
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches, matched_pred, matched_gt = {}, set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] == 1e10:
            continue
        matches[pred_ids[r]] = {
            "gt_match": gt_ids[c],
            "pred_time": pred_times[r],
            "gt_time": gt_times[c],
            "time_diff_min": cost_matrix[r, c]
        }
        matched_pred.add(pred_ids[r])
        matched_gt.add(gt_ids[c])

    return matches, matched_pred, matched_gt


def summarise_match_criteria(df_analyses, criteria):
    """
    Prints how the match sets produced by each of `criteria` compare.

    `criteria` is the list get_matching_cmes builds, so adding a criterion there shows up here
    without further changes. Per criterion this reports how many CMEs it matched and the spread
    of its own time difference; then, for each pair of criteria, how many CMEs both of them
    matched, how many of those they assigned to the same STRUDL track, and how many each matched
    alone. Criteria whose columns are absent -- a criterion that was skipped -- are left out.

    The per-criterion time differences are each measured against that criterion's own pair of
    times, so they say how well a criterion agrees with itself, not which criterion is better.
    What is comparable across criteria is the match counts and the agreement.
    """

    present = [criterion for criterion in criteria if criterion['id_column'] in df_analyses.columns]

    if not present:
        print("No matching criteria produced any matches.")
        return

    print("\nMatch criteria comparison")
    print("-" * 78)

    for criterion in present:
        matched = df_analyses[df_analyses[criterion['id_column']].notna()]
        differences = pd.to_numeric(matched[criterion['diff_column']], errors='coerce')
        print(f"  {criterion['name']:<22} {len(matched):>4} matches   "
              f"|dt| median {differences.median():>6.0f} min, "
              f"IQR {differences.quantile(0.25):>5.0f}-{differences.quantile(0.75):<5.0f} "
              f"(threshold {criterion['threshold']})")

    for first, second in itertools.combinations(present, 2):
        ids_first = df_analyses[first['id_column']]
        ids_second = df_analyses[second['id_column']]
        both = ids_first.notna() & ids_second.notna()
        agree = (ids_first[both] == ids_second[both]).sum()
        print(f"\n  {first['name']} vs {second['name']}:")
        print(f"    matched by both {both.sum()}, of which they agree on the same track {agree} "
              f"and disagree on {both.sum() - agree}")
        print(f"    matched only by {first['name']}: {(ids_first.notna() & ids_second.isna()).sum()}, "
              f"only by {second['name']}: {(ids_second.notna() & ids_first.isna()).sum()}")

    print("-" * 78)


def get_matching_cmes(df_pred, df_analyses, threshold=60*6, threshold_fpf=60*3,
                      threshold_original=60*6, criterion_column='couldBeInHI1AFov'):
    """
    Matches STRUDL tracks to DONKI CMEs under three criteria, reporting all of them.

    1. 'HI-1A entry' -- STRUDL's `first_hi_time`, the time its front was first picked up in
       HI-1A, against DONKI's `predictedFirstHI1Atime`, the time its CME's propagated ellipsoid
       first reaches the HI-1A FOV.

    2. 'original' -- `first_hi_time` against `time21_5`, the criterion used before any of this.

    3. 'FPF launch' -- the launch time of an unconstrained Fixed-Phi fit to the STRUDL track cut
       at its central position angle, against DONKI's own`associatedCMEstartTime`.

    Criterion 3 needs df_pred to have been through add_fpf_fits; without it, only 1 and 2 run.
    """
    identifier_ground_truth = 'associatedActivityID'
    identifier_predicted = 'track_key'

    df_pred['first_hi_time'] = pd.to_datetime(df_pred['first_hi_time'], format='%Y-%m-%d %H:%M:%S')
    for time_column in ('predictedFirstHI1Atime', 'time21_5', 'associatedCMEstartTime'):
        df_analyses[time_column] = pd.to_datetime(df_analyses[time_column].values,
                                                  format='%Y-%m-%dT%H:%MZ', utc=True)

    df_gt = df_analyses[df_analyses[criterion_column]]

    criteria = [
        {'name': 'HI-1A entry', 'match_key_pred': 'first_hi_time',
         'match_key_gt': 'predictedFirstHI1Atime', 'threshold': threshold,
         'id_column': 'associatedStrudlID', 'time_column': 'strudlFirstHItime',
         'diff_column': 'timeDifferenceMinutes', 'threshold_column': 'thresholdMinutes'},
        {'name': 'original (time21_5)', 'match_key_pred': 'first_hi_time',
         'match_key_gt': 'time21_5', 'threshold': threshold_original,
         'id_column': 'associatedStrudlIDOriginal', 'time_column': 'strudlFirstHItimeOriginal',
         'diff_column': 'timeDifferenceOriginalMinutes', 'threshold_column': 'thresholdOriginalMinutes'},
        {'name': 'FPF launch', 'match_key_pred': 'fpf_launch_time',
         'match_key_gt': 'associatedCMEstartTime', 'threshold': threshold_fpf,
         'id_column': 'associatedStrudlIDFpf', 'time_column': 'strudlFpfLaunchTime',
         'diff_column': 'timeDifferenceFpfMinutes', 'threshold_column': 'thresholdFpfMinutes'},
    ]

    # the Fixed-Phi fit parameters of whichever track criterion 3 picked, carried along so a
    # questionable fit can be told from a sound one without going back to the track
    fpf_diagnostics = {'strudlFpfPhi': 'fpf_phi_deg', 'strudlFpfSpeed': 'fpf_speed_kms',
                       'strudlFpfRmsDeg': 'fpf_rms_deg', 'strudlFpfOnBound': 'fpf_on_bound',
                       'strudlFpfNPoints': 'fpf_n_points', 'strudlFpfCentralPA': 'fpf_central_pa'}

    for criterion in criteria:
        if criterion['match_key_pred'] not in df_pred.columns:
            print(f"Skipping the {criterion['name']} criterion: "
                  f"df_pred has no '{criterion['match_key_pred']}' column")
            continue

        matches, _, _ = match_cmes(df_pred,
                                    df_gt,
                                    match_key_ground_truth=criterion['match_key_gt'],
                                    match_key_predicted=criterion['match_key_pred'],
                                    identifier_ground_truth=identifier_ground_truth,
                                    identifier_predicted=identifier_predicted,
                                    threshold=criterion['threshold'])

        is_fpf = criterion['match_key_pred'] == 'fpf_launch_time'
        fits = df_pred.set_index(identifier_predicted) if is_fpf else None

        for pred_id, match_info in matches.items():
            matching_row_analyses = df_analyses[identifier_ground_truth] == match_info['gt_match']
            df_analyses.loc[matching_row_analyses, criterion['id_column']] = pred_id
            df_analyses.loc[matching_row_analyses, criterion['time_column']] = np.datetime_as_string(match_info['pred_time'], unit='m', timezone='UTC')
            df_analyses.loc[matching_row_analyses, criterion['diff_column']] = match_info['time_diff_min']
            df_analyses.loc[matching_row_analyses, criterion['threshold_column']] = criterion['threshold']

            if is_fpf:
                for out_column, fit_column in fpf_diagnostics.items():
                    df_analyses.loc[matching_row_analyses, out_column] = fits.loc[pred_id, fit_column]

        print(f"{criterion['name']} criterion: {len(matches)} matches "
              f"at a {criterion['threshold']} minute threshold")

    summarise_match_criteria(df_analyses, criteria)

    # to_json defaults to epoch timestamps for datetime columns, so put the times back to the
    # '%Y-%m-%dT%H:%MZ' strings they came in as
    for time_column in ('predictedFirstHI1Atime', 'time21_5', 'associatedCMEstartTime'):
        df_analyses[time_column] = df_analyses[time_column].apply(
            lambda x: None if pd.isna(x) else np.datetime_as_string(x.to_numpy(), unit='m', timezone='UTC'))

    return df_analyses
