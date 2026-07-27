import os
import datetime
import requests
import pandas as pd
import numpy as np
import sys
from sunpy.coordinates.ephemeris import get_horizons_coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from sunpy.coordinates import Helioprojective, HeliographicStonyhurst
import pyvista as pv
from scipy.optimize import linear_sum_assignment
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst

geosphere_purple = '#701F80'
geosphere_green = '#75CC41'
geosphere_yellow = '#F9F200'
geosphere_orange = '#F29707'
my_pink = '#FF69B4'

def download_donki_data(file_path, data_type):
    """
    Download DONKI data for the specified data type and save it as a JSON file.

    Parameters:
    - file_path: str, the path where the JSON file will be saved.
    - data_type: str, the type of DONKI data to download (e.g., 'CME', 'IPS', etc.).

    """

    data_dir = os.path.dirname(file_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Get today's date in the required format
    date_today = datetime.datetime.now().strftime("%Y-%m-%d")
    url_donki = 'https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/'+data_type+'?startDate=2000-01-01&endDate='+date_today

    try:
        response = requests.get(url_donki, timeout=600)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx/5xx)

        # Save content as JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download DONKI data: {e}")

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

    return analyses_df

def convert_strudl_output(path_to_strudl_dict, pa_array=None, use_earth_pa=True, use_max_elon_pa=False, sc='A'):
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
    sc : str, default='A'
        Spacecraft identifier to use when calculating Earth's position angle.
    Returns
    -------

    ------
    SystemExit
        If both `pa_array` and `use_earth_pa` or `use_max_elon_pa` are set, if the length of `pa_array` does not match the number
        of CMEs, or if neither `pa_array` nor `use_earth_pa` or `use_max_elon_pa` are set.
    """
    
    strudl_dict = np.load(path_to_strudl_dict, allow_pickle=True).item()
    
    if pa_array is not None and (use_earth_pa or use_max_elon_pa):
        sys.exit("Error: Both pa_array and use_earth_pa/use_max_elon_pa are set.")

    elif pa_array is not None and len(pa_array) != len(strudl_dict):
        sys.exit("Error: Length of pa_array does not match number of CMEs in STRUDL dictionary.")
    
    elif pa_array is None and not (use_earth_pa or use_max_elon_pa):
        sys.exit("Error: Neither pa_array nor use_earth_pa/use_max_elon_pa are set.")

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

def get_stereo_heeq_lon(pos_time, sc):
    """
    Get the Heliocentric Earth Equatorial (HEEQ) longitude of a STEREO spacecraft at a given time.
    Parameters
    ----------
    pos_time : astropy.time.Time or str
        The time at which to obtain the STEREO spacecraft position. Can be an astropy Time object or a string parseable by astropy.
    sc : str
        The STEREO spacecraft identifier. Must be either 'A' for STEREO-A or 'B' for STEREO-B.
    Returns
    -------
    float
        The longitude of the STEREO spacecraft in Heliocentric Earth Equatorial (HEEQ) coordinates, in degrees.
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
    stereo_heeq_lon = np.array([SkyCoord(pos).transform_to(HeliographicStonyhurst()).lon.to(u.deg).value for pos in stereo_pos])
    
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

def make_circle(radius,center_x, center_y,num_points ):
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
    ex_cart, ey_cart, ez_cart = sphere_to_cart_heeq(ex, ey, ez)
    Earth = pv.Sphere(radius=1.0*r_sun,center=(ex_cart, ey_cart, ez_cart))

    # STEREO-A
    if plot_stereo:

        stereo_x, stereo_y, stereo_z = [stereo_coords.radius.value*AU_in_Rsun, np.deg2rad(stereo_coords.lat.value), np.deg2rad(stereo_coords.lon.value)]
        stereo_x_cart, stereo_y_cart, stereo_z_cart = sphere_to_cart_heeq(stereo_x, stereo_y, stereo_z)
        sta = pv.Sphere(radius=1.0*r_sun,center=(stereo_x_cart, stereo_y_cart, stereo_z_cart))

    X, Y, Z = ex_cart, ey_cart, ez_cart 

    ecliptic_inclination = np.deg2rad(7.25) 

    n_z = np.cos(ecliptic_inclination)  
    n_x = -n_z * Z / X
    n_y = np.sqrt(1- n_x**2 - n_z**2) # sqrt(nx**2 + ny**2 +nz**2) = 1 -> nx**2 + ny**2 +nz**2 = 1**2 -> ny = sqrt(1-nx**2-nz**2) -> ny = sqrt(1-nx**2-cos(ecliptic_inclination)**2) -> ny = sqrt(sin(ecliptic_inclination)-nx**2)

    if Y<0:
        n_y = -n_y
    else:
        n_y = n_y
    
    ecliptic_normal = np.array([n_x, n_y, n_z])

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
            min_lon_intersection  = np.min(lon_farenough)
            max_lon_intersection  = np.max(lon_farenough)  

            max_angle = (max_lon_intersection-min_lon_intersection)/2.0
            apex_long = min_lon_intersection+max_angle

            idx_closest_to_mean = np.argsort(np.abs(lon-lon_intersection))[0]
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
                        label = pv.Label(str(i*10.0)+" R☉", position=tip,size=18)
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
                plotter.add_mesh(lemniscate_mesh.extract_surface(), color=color_lemniscate,opacity=0.35, show_edges=True)

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
    
def convert_strudl_output_start_time_only(path_to_strudl_dict, sc='A'):
    """
    Converts the output from a STRUDL dictionary file into a track dictionary, optionally using a provided
    position angle (PA) array or calculating the Earth's position angle for each CME.
    Parameters
    ----------
    path_to_strudl_dict : str
        Path to the .npy file containing the STRUDL dictionary with CME data.
    sc : str, default='A'
        Spacecraft identifier to use when calculating Earth's position angle.

    Outputs
    -------
    Saves a numpy file with the start times of each CME in the STRUDL dictionary.
    """
    
    # loop over every CME in STRUDL dictionary and cut along desired position angle
    strudl_dict = np.load(path_to_strudl_dict, allow_pickle=True).item()
    strudl_times = [[key, 'strudl', strudl_dict[key]['times'][0], sc] for key in strudl_dict.keys()]
    track_df = pd.DataFrame(strudl_times, columns=['track_key', 'track_no', 'first_hi_time', 'sc'])

    return track_df

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

    # Apply threshold (minutes); set large cost where invalid
    cost_matrix[np.logical_or(cost_matrix > threshold, np.isnan(cost_matrix))] = 1e10

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


def get_matching_cmes(df_pred, df_analyses, threshold=60*6):
    """
    """
    match_key_gt = 'time21_5'
    match_key_pred = 'first_hi_time'

    identifier_ground_truth = 'associatedActivityID'
    identifier_predicted = 'track_key'

    feature_codes = df_analyses['featureCode'].unique().tolist()

    df_pred[match_key_pred] = pd.to_datetime(df_pred[match_key_pred], format='%Y-%m-%d %H:%M:%S')
    df_analyses[match_key_gt] = pd.to_datetime(df_analyses[match_key_gt].values, format='%Y-%m-%dT%H:%MZ', utc=True)

    matches_all = {code: [] for code in feature_codes}
    for code in feature_codes:
        print(f"Processing feature code: {code}")
        
        df_gt = df_analyses[df_analyses['featureCode'] == code]
        #df_gt = df_gt[df_gt['hasEarthArrival']]
        df_gt = df_gt[df_gt['arrivalEllipsePossibleCorrected']]

        # match CMEs based on time difference of starting time in HI / time at 21.5 Rsun
        matches, _, _ = match_cmes(df_pred,
                                    df_gt,
                                    match_key_ground_truth=match_key_gt,
                                    match_key_predicted=match_key_pred,
                                    identifier_ground_truth=identifier_ground_truth,
                                    identifier_predicted=identifier_predicted,
                                    threshold=threshold)
        
        matches_all[code] = matches

    for code, matches in matches_all.items():
        for pred_id, match_info in matches.items():
            gt_id = match_info['gt_match']
            matching_row_analyses = (df_analyses[identifier_ground_truth] == gt_id) & (df_analyses['featureCode'] == code)
            df_analyses.loc[matching_row_analyses, 'associatedStrudlID'] = pred_id
            df_analyses.loc[matching_row_analyses, 'strudlFirstHItime'] = np.datetime_as_string(match_info['pred_time'], unit='m', timezone='UTC')
            df_analyses.loc[matching_row_analyses, 'timeDifferenceMinutes'] = match_info['time_diff_min']
            df_analyses.loc[matching_row_analyses, 'thresholdMinutes'] = threshold

    df_analyses['time21_5'] = df_analyses['time21_5'].apply(lambda x: np.datetime_as_string(x.to_numpy(), unit='m', timezone='UTC'))
    return df_analyses