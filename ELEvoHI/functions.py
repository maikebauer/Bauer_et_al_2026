import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from datetime import datetime,timedelta
import matplotlib.dates as mdates
from astropy.time import Time as atime
from astropy import units as u
import matplotlib.patheffects as path_effects
from astropy import constants as const
import csv
import pandas as pd
import json
import pickle
import copy
from numba import njit, prange
from astropy import constants as const

# Constants
AU = const.au.to_value('km')
rsun = const.R_sun.to_value('km')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def calculate_new_time_axis(start_time, end_time, cadence):
    # Function to calculate new time axis
    timerange =  pd.date_range(start=start_time, end=end_time+timedelta(minutes=cadence//4), freq=str(cadence)+"min")
    return timerange

def fpf_function(xdata, phi, speed):
    # function to fit time-elongation profile of CME
    # assumptions: constant speed, fixed direction
    elon_fit = np.arctan((speed * xdata * np.sin(phi)) / (1. - speed * xdata * np.cos(phi)))

    return elon_fit

def fpf_function_aswo(xdata, phi, speed, launch_init):
    # function to fit time-elongation profile of CME
    # assumptions: constant speed, fixed direction

    scrad = 1.0

    xdata_adjusted  = xdata - launch_init

    elon_fit = np.arctan((speed * xdata_adjusted * np.sin(phi)) / (scrad - speed * xdata_adjusted * np.cos(phi)))
    
    return elon_fit

def fpf_mab(track, startcut, endcut, prediction_path):

    df = track.copy()
    
    # Slice elongation
    elon = df["elongation"][startcut:endcut]
    
    # Slice time
    time_nu = np.array(df["time"][startcut:endcut])
    
    # time for plotting
    time = df["time"][startcut:endcut]
    
    # Convert to astropy Time objects
    t_utc = atime(time_nu, scale="utc")

    # Convert to TAI
    t_tai = t_utc.tai

    # Seconds relative to the first point in the slice
    taitime = (t_tai - t_tai[0]).sec

    # Normalized time axis
    time_num = taitime - np.min(taitime)

    # Output
    xdata = time_num
    ydata = elon
    
    # Initial guess for parameters phi and speed
    phi_init = np.deg2rad(60.)
    speed_init = 500./AU 
    launch_offset_init=-20.0*3600.0
    
    bounds = ([0, 200./AU, launch_offset_init], [np.pi, 4000./AU, 0.])  # Example bounds: phi between 0 and pi/2, speed >= 0
    
    ydata_rad = np.deg2rad(ydata)
    
    # Perform the curve fitting

    try:

        params, covariance = curve_fit(fpf_function_aswo, xdata, ydata_rad, p0=[phi_init, speed_init, launch_offset_init], bounds=bounds)
        phi_fit, speed_fit, tinit_fit = params     
        
    except ValueError:
    
        return np.nan, np.nan, [np.nan]*len(xdata)
    
   
    
    # Generate the fitted curve using the fitted parameters
    elon_fit = fpf_function_aswo(xdata, phi_fit, speed_fit, tinit_fit)
    
    # Absolute launch time from tinit_fit
    launch_time_fit = t_tai[0] + tinit_fit * u.s  # tinit_fit in seconds, t_tai[0] is an Astropy Time object

    # Arrival time at 1 AU
    #travel_time = AU / speed_fit  # in seconds
    travel_time = 1.0 / speed_fit * u.s
    arrival_time_fit = launch_time_fit + travel_time

    # Convert to datetime for display
    formatted_launch = launch_time_fit.datetime.strftime("%Y-%m-%dT%H:%M")
    formatted_arrival = arrival_time_fit.datetime.strftime("%Y-%m-%dT%H:%M")

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')

    ax.scatter(time, ydata, label="Data")
    ax.plot(time, np.rad2deg(elon_fit), label="Fixed-Phi Fit")

    ax.set_xlabel("Time (t)", fontsize=14)
    ax.set_ylabel("Elongation [°]", fontsize=14)
    ax.set_title("Fixed-Phi Fit", fontsize=16)

    # Format the x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.legend(fontsize=14)

    # Text annotations for the plot
    p_rounded = round(np.rad2deg(phi_fit))
    s_rounded = round(speed_fit * AU)
    

    results_text = 'Fitting results:'
    phi_text = f'$\\phi$ = {p_rounded}°'
    speed_text = f'V$_{{const.}}$ = {s_rounded} km s$^{{-1}}$'

    underlined_effect = [path_effects.SimpleLineShadow(), path_effects.Normal()]

    text_loc = round(len(time) * 0.7)
    step = (round(max(ydata) - min(ydata))) / 10

    # Coordinates for placing the text annotations
    results_x, results_y = time[text_loc], round(max(ydata) - min(ydata)) * 0.5 + round(min(ydata))
    speed_x, speed_y = results_x, results_y - step
    phi_x, phi_y = results_x, results_y - 2*step

    ax.text(results_x, results_y, results_text, fontsize=16, path_effects=underlined_effect)
    ax.text(phi_x, phi_y, phi_text, fontsize=14)
    ax.text(speed_x, speed_y, speed_text, fontsize=14)

    # Create output directory if it doesn't exist
    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)

    plt.savefig(prediction_path + 'FPF_mab.png', dpi=150, bbox_inches='tight')
    plt.clf()
    plt.close()

    # Save results as a pickle file
    fpf_results = {
        'launch_time_FPF': launch_time_fit.datetime,
        'phi_FPF': p_rounded,
        'speed_FPF': s_rounded,
        'arrivaltime_L1_FPF': arrival_time_fit.datetime
    }

    with open(prediction_path + 'FPF_results.pkl', 'wb') as file:
        pickle.dump(fpf_results, file)

    return fpf_results

    
def fpf(track, startcut, endcut, prediction_path):
    
    #df = pd.read_csv(track_data)
    df = copy.deepcopy(track)
    elon = df["elongation"][startcut:endcut]
    std = df["std"][startcut:endcut]

    # Convert the "time" column to datetime objects
    #df["time"] = pd.to_datetime(df["time"])

    time_nu = df["time"][startcut:endcut]

    # Convert the 'time' column to a numerical time array
    time_n = time_nu.apply(lambda x: x.timestamp())

    # calculate launch time of CME
    launch_elon = 0

    # Create an interpolating function to extrapolate time_num based on elon down to `launch_elon`
    f = interp1d(elon, time_n, fill_value="extrapolate")

    # Extrapolate time_num up to `desired_elon`
    launch_time_n = f(launch_elon)
    launch_time = np.datetime64('1970-01-01') + np.array(launch_time_n, dtype='timedelta64[s]')

    time = df["time"][startcut:endcut]

    #fixed_timestamp = time.iloc[0] - pd.DateOffset(launch_time_num)

    # Subtract the fixed timestamp from each element in the 'time' array
    time_num = (time - launch_time).dt.total_seconds()

    xdata = time_num
    ydata = elon
    
    
    # Initial guess for parameters phi and speed
    phi_init = np.deg2rad(60.)
    speed_init = 500./AU 

    bounds = ([0, 200./AU], [np.pi, 4000./AU])  # Example bounds: phi between 0 and pi/2, speed >= 0

    ydata_rad = np.deg2rad(ydata)

    # Perform the curve fitting
    params, covariance = curve_fit(fpf_function, xdata, ydata_rad, p0=[phi_init, speed_init], bounds=bounds)

    # Extract the fitted parameters
    phi_fit, speed_fit = params

    # Generate the fitted curve using the fitted parameters
    elon_fit = fpf_function(xdata, phi_fit, speed_fit)

    # Create the plot and get the axes object
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')  # Adjust figure size as needed

    # Plot the data and fitted curve
    ax.scatter(time, ydata, label="Data")
    ax.plot(time, np.rad2deg(elon_fit), label="Fixed-Phi Fit")

    # Set the x-axis label
    ax.set_xlabel("Time (t)", fontsize=14)

    # Set the y-axis label
    ax.set_ylabel("Elongation [°]", fontsize=14)

    # Set the title
    ax.set_title("Fixed-Phi Fit", fontsize=16)

    # Format the datetime axis labels
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    # Customize the x-axis ticks for 12-hour intervals
    hour_locator = mdates.HourLocator(interval=12)
    ax.xaxis.set_major_locator(hour_locator)

    ax.legend(fontsize=14)

    p_rounded = round(np.rad2deg(phi_fit))
    s_rounded = round(speed_fit * AU)

    results_text = 'Fitting results:'
    phi_text = f'$\\phi$ = {p_rounded}°'
    speed_text = f'V$_{{const.}}$ = {s_rounded} km s$^{{-1}}$'

    underlined_effect = [path_effects.SimpleLineShadow(),
                        path_effects.Normal()]
    text_loc = round(len(time)*0.7)
    
    step = (round(max(ydata) - min(ydata)))/10

    # Coordinates for placing the text annotations
    results_x, results_y = time[text_loc], round(max(ydata) - min(ydata)) * 0.5 + round(min(ydata))
    speed_x, speed_y = time[text_loc], round(max(ydata) - min(ydata)) * 0.5 + round(min(ydata)) - step
    phi_x, phi_y = time[text_loc], round(max(ydata) - min(ydata)) * 0.5 + round(min(ydata)) - 2*step

    # Text annotations to the plot
    ax.text(results_x, results_y, results_text, fontsize=16, path_effects=underlined_effect)
    ax.text(phi_x, phi_y, phi_text, fontsize=14)
    ax.text(speed_x, speed_y, speed_text, fontsize=14)

    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)

    plt.savefig(prediction_path+'FPF_tam.png', dpi=150, bbox_inches='tight')
    
    plt.clf()
    plt.close()
    
    # Arrival time at 1 AU

    travel_time = AU/s_rounded

    FPF_Earth_arrival_np = np.datetime64('1970-01-01') + np.array((launch_time_n + travel_time), dtype='timedelta64[s]')
    # Convert FPF_Earth_arrival_np to a regular datetime object
    FPF_Earth_arrival = FPF_Earth_arrival_np.astype(datetime)
    formatted_arrival = FPF_Earth_arrival.strftime("%Y-%m-%dT%H:%M")
    launch = launch_time.astype(datetime)
    formatted_launch = launch.strftime("%Y-%m-%dT%H:%M")

    # save results 

    fpf_results = {
        'launch_time_FPF': launch_time,
        'phi_FPF': round(np.rad2deg(phi_fit)),
        'speed_FPF': round(speed_fit * AU),
        'arrivaltime_L1_FPF': FPF_Earth_arrival
    }

    with open(prediction_path + 'FPF_results.pkl', 'wb') as file:
        pickle.dump(fpf_results, file)
    
    return fpf_results

def ELCon(elon, d, phi, hwidth, f):
    """ ELlipse Conversion method (Rollett et al., 2016)

    converts HI time-elongation measurements to time-distance.
    """
    p = phi
    l = hwidth
    e = elon
    R_ell = np.zeros(np.size(elon))

    for i in range(len(elon)):
        
        beta1 = np.pi - e[i] - p

        if beta1 > np.pi/2.:
            beta1 = e[i] + p

        theta = np.arctan(f**2 * np.tan(beta1))
        w = np.sqrt((np.cos(theta)**2) * (f**2 - 1) + 1)

        thetas = np.arctan(f**2 * np.tan(l))
        ws = np.sqrt((np.cos(thetas)**2) * (f**2 - 1) + 1)

        X = ((np.cos(l-thetas)/np.sin(l)) + ws)**(-1) * ((np.cos(e[i]+p+theta)/(w*np.sin(e[i]+p))) + 1)

        if (np.pi-e[i]-p) > np.pi/2:
            X = ((np.cos(l-thetas)/np.sin(l))+ws)**(-1) * ((-np.sin(np.pi/2+theta-e[i]-p)/(w*np.sin(e[i]+p))) + 1)

        Y = (d * np.sin(e[i]))/(np.sin(e[i] + p))

        R_ell[i] = Y/(1 - ws*X)
        
    return R_ell

def fitdbm(x, gamma, vinit, swspeed, rinit):
    """ function to fit time-distance profile of CME """   
    fit = (1/gamma) * np.log(1 + gamma*(vinit - swspeed) * x) + swspeed*x + rinit
    
    return fit

def fitdbmneg(x, gamma, vinit, swspeed, rinit):
    """ function to fit time-distance profile of CME """   
    fit = (-1/gamma) * np.log(1 - gamma*(vinit - swspeed) * x) + swspeed*x + rinit
    
    return fit

def logistic_growth(xdata, k):
    return xdata.shape[0]/(1+np.exp(
                                    -k*(
                                        np.arange(1,xdata.shape[0]+1,1)-xdata.shape[0]//2
                                        )
                                    )
                            )

def cost_function(params, *args):
    gamma = params
    vinit, swspeed, rinit, ydata, x = args
    predicted = fitdbm(x, gamma, vinit, swspeed, rinit)

    return np.mean(np.abs(ydata - predicted))# * logistic_growth(ydata,0.3)/logistic_growth(ydata,0.3).max() )

    return np.mean(np.abs(ydata - predicted))# * logistic_growth(ydata,0.3)/logistic_growth(ydata,0.3).max() )


def cost_functionneg(params, *args):
    gamma = params
    vinit, swspeed, rinit, ydata, x = args
    predicted = fitdbmneg(x, gamma, vinit, swspeed, rinit)
    #np.median(np.sqrt((ydata - predicted) ** 2 ))
    return np.mean(np.abs(ydata - predicted))

def cost_function_updated(params, *args):
    gamma, vinit = params
    swspeed, rinit, ydata, x = args
    predicted = fitdbm(x, gamma, vinit, swspeed, rinit)

    return np.mean(np.abs((ydata - predicted)))# * logistic_growth(ydata,0.3)/logistic_growth(ydata,0.3).max() )

def cost_functionneg_updated(params, *args):
    gamma, vinit = params
    swspeed, rinit, ydata, x = args
    predicted = fitdbmneg(x, gamma, vinit, swspeed, rinit)

    return np.mean(np.abs((ydata - predicted)))

# def cost_functionneg(gamma):
#     predicted = fitdbmneg(xdata, gamma)
#     return np.sum((ydata - predicted) ** 2)

def DBMfitting(time, distance_au, prediction_path, det_plot, startfit = 1, endfit = 20, silent = 1, max_residual = 1.5, max_gamma = 2e-7):
    """ fit the ELCon time-distance track using the drag-based equation of motion from Vrsnak et al. (2013) """
    global  runnumber

    # start end end points of fitting.
    # This was done manually so far, but should be implemented in a way that it is done automatically.
    # However, it might always be necessary that a forecaster reviews the fitting because not each HI kinematics can
    # be fitted from each startpoint on.

    # startfit = 3 -> default settings
    # endfit = 18 -> default settings
    # Convert datetime values to seconds since the first element
    time_num = (time - time[0]).total_seconds()
    
    speedtime = time_num - time_num[0]
    distance_km = distance_au * AU
    distance_rsun = distance_au * AU / rsun

    #build time derivative from ELCon CME time-distance track
    speed = np.gradient(distance_km, speedtime)

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #!!! How do we build the speed errors? In IDL this is done by the function DERIVSIG.
    # In Python, I have no idea! -> This needs to be solved.
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    tinit = time[startfit]
    rinit = distance_km[startfit]
    vinit = speed[startfit]

    xdata = speedtime.values
    ydata = distance_km

    #testgamma = 1.7863506087704678e-08 
    #testwind  = 200

    #testy = (1/testgamma) * np.log(1 + testgamma*(vinit - testwind) * xdata) + testwind*xdata + rinit

    winds = np.arange(250, 775, 25)
    fit = np.zeros((len(winds), len(xdata)))
    fitspeed = np.zeros((len(winds), len(xdata)))
    residuals = np.zeros((len(winds), len(xdata)))
    gamma = np.zeros(len(winds))
    success = np.zeros(len(winds))
    
    # Initial guess for the parameter gamma
    initial_guess = 1e-7
    k= 0.3

    # do a fit for each wind speed separately and plot valid fits
    for i in range(len(winds)):
        swspeed = winds[i]
        if vinit > swspeed:
            # Perform the optimisation
            # Perform the optimisation
            result = minimize(cost_function, initial_guess,args=(vinit,swspeed,rinit,ydata,xdata), method='Nelder-Mead')
            # gamma_fit, pcov = curve_fit(fitdbm, xdata, ydata,p0=initial_guess,method="dogbox")
            # Print the fitted parameter

            success[i] = True
            if result.success:
                gamma_fit = result.x[0]

                # fit_ = fitdbm(xdata, gamma_fit)
                fit_ = fitdbm(xdata, gamma_fit, vinit, swspeed, rinit)
                gamma[i] = gamma_fit
                fit[i,:] = fit_   
                residuals[i,:] = np.abs(ydata - fit_)#np.median(np.sqrt((ydata - fit_) ** 2 ))# * logistic_growth(ydata,k)/logistic_growth(ydata,k).max())
                fitspeed[i,:] = np.gradient(fit[i,:], xdata)
            else:
                continue

        else:
            # Perform the optimisation
            result = minimize(cost_functionneg, initial_guess, args=(vinit,swspeed,rinit,ydata,xdata), method='Nelder-Mead')

            success[i] = result.success
            # Print the fitted parameter
            if result.success:
                gamma_fit = result.x[0]

                fit_ = fitdbmneg(xdata, gamma_fit, vinit, swspeed, rinit)
                gamma[i] = gamma_fit
                fit[i,:] = fit_   
                residuals[i,:] = np.abs(ydata - fit_)#np.median(np.abs(ydata - fit_)) test because the wrong wind speed is chosen
                fitspeed[i,:] = np.gradient(fit[i,:], xdata)

            else:
                continue
            
    res = np.zeros(len(winds))

    for i in range(len(winds)):
        if success[i]:
            res[i] = np.mean(residuals[i,:])#residuals[i,0] #np.mean(residuals[i,:])
        else:
            res[i] = np.nan
            
    if det_plot:        
        sns.set_context('talk')
        sns.set_style('darkgrid')
        figdbm, ax = plt.subplots(1, 1, figsize = (8,3), dpi = 300, facecolor='white')
        
    if det_plot:        
        sns.set_context('talk')
        sns.set_style('darkgrid')
        figdbm, ax = plt.subplots(1, 1, figsize = (8,3), dpi = 300, facecolor='white')
        

        ax.set_title('ELEvoHI DBMfits', size = 16)
        ax.set_ylabel('CME Apex Speed [km s$^{-1}$]', fontsize = 14, labelpad = 0)
        ax.set_xlabel('Heliocentric Distance [R$_\odot$]', fontsize = 14, labelpad = 0)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 10, pad = -5)

        norm = mpl.colors.Normalize(vmin = winds.min(), vmax = winds.max())
        cmap = mpl.cm.ScalarMappable(norm = norm, cmap = mpl.cm.jet)
        cmap.set_array([])

        ax = plt.gca()
        cbar = plt.colorbar(cmap, pad = 0.02, ax=ax)
        cbar.set_label('solar wind speed [km s$^{-1}$]', rotation = 270, fontsize = 14, labelpad = 20)
        cbar.ax.tick_params(axis='y', direction='inout', length=5, width=0, labelsize=12)
    
    gamma_v = []
    res_v = []
    winds_v = []

    residuals2 = []
    for i in range(len(winds)):
        if success[i] != True:
            continue
        if res[i]/rsun > max_residual:
            continue
        if gamma[i] <= 0:
            continue
        if max_gamma >= np.abs(gamma[i]):
            gamma_v.append(gamma[i])
            res_v.append(res[i]/rsun)
            winds_v.append(winds[i])
            if det_plot:
                ax.plot(distance_rsun, fitspeed[i,:], label='fit', linewidth=1, c=cmap.to_rgba(winds[i]))
        else:
            continue

    if det_plot:
        ax.plot(distance_rsun, speed, 'o', label='data', c='black', markersize=5)
        
        elcon_out = pd.DataFrame({
                "time": time,
                "R": distance_rsun,
                "V": speed
            })
        
        elcon_out.to_csv(prediction_path + "ELCon.csv", index=False, header=["time", "heliocentric distance", "speed"])

        # Calculate y-axis limits
        # y_lower = min(speed) - 100
        # y_upper = max(speed) + 100
        
        # Use fixed y-axis limits
        y_lower = 0
        y_upper = 1500
        y_upper = 1500

        # Set the y-axis limits
        ax.set_ylim(y_lower, y_upper)
    
    # find the most accurate fitting parameters based on the minimum mean residual
    
    if len(gamma_v) != 0:
        gamma_valid = np.array(gamma_v)   
        res_valid = np.array(res_v)   
        winds_valid = np.array(winds_v)   
        min_res = np.argmin(np.abs(res_valid))
    else:
        gamma_valid = [0]
        winds_valid = [0]
        res_valid = [0]
        tinit = 0
        rinit = 0
        vinit = 0
        swspeed = 0
        xdata = [0]
        ydata = [0]
        speed = [0]
        return gamma_valid, winds_valid, res_valid, tinit, rinit, vinit, swspeed, xdata, ydata, speed

    # Get the indices that would sort the array based on absolute values
    sorted_indices = np.argsort(np.abs(res_valid))
    gamma_valid = gamma_valid[sorted_indices]
    res_valid = res_valid[sorted_indices]
    winds_valid = winds_valid[sorted_indices]
  
    #pdb.set_trace()
    
    if det_plot:
        filename = prediction_path + '/DBMfit.png' 
        plt.savefig(filename, dpi=300, facecolor=figdbm.get_facecolor(), edgecolor='none', bbox_inches='tight')       
        figdbm.clf()
        plt.close(figdbm)

    
    return gamma_valid, winds_valid, res_valid, tinit, rinit, vinit, swspeed, xdata, ydata, speed


def DBMfitting_updated(time, distance_au, prediction_path, det_plot, startfit = 0, silent = 1, max_residual = 1.5, max_gamma = 2e-7, compute_vinit='first', **kwargs):
    """ fit the ELCon time-distance track using the drag-based equation of motion from Vrsnak et al. (2013).
        This is an updated version of the DBMfitting function.
        Added:
        - use mean of first 50 RSun for starting speed, otherwise first value
        - use boundaries for dbm fitting (minimize)
    """
    global  runnumber

    # compute_vinit: first, mean_points, mean_rsun, mean_donki_cat
    # start end end points of fitting.
    # This was done manually so far, but should be implemented in a way that it is done automatically.
    # However, it might always be necessary that a forecaster reviews the fitting because not each HI kinematics can
    # be fitted from each startpoint on.

    # startfit = 3 -> default settings
    # endfit = 18 -> default settings
    # Convert datetime values to seconds since the first element

    required_by_method_vinit = {
        "first": set(),
        "mean_points": {"num_points"},
        "mean_rsun": {"cutoff_rsun"},
        "mean_donki_cat": {"vinit_input"}
    }

    if compute_vinit not in required_by_method_vinit:
        raise ValueError(f"Unknown method: {compute_vinit}")

    required_kws = required_by_method_vinit[compute_vinit]
    missing_kws = required_kws - kwargs.keys()

    if missing_kws:
        raise ValueError(f"Method '{compute_vinit}' requires parameters: {', '.join(sorted(missing_kws))}")
    
    distance_rsun = distance_au[startfit:] * AU / rsun
    time_num = (time[startfit:] - time[startfit]).total_seconds()

    speedtime = time_num - time_num[0]
    distance_km = distance_au[startfit:] * AU
    speed = np.gradient(distance_km, speedtime)

    if compute_vinit == 'first':
        vinit_estimate = speed[0]
        rinit = distance_km[0]
        tinit = time[0]

        xdata = speedtime.values
        ydata = distance_km

    elif compute_vinit == 'mean_points':
        num_points = kwargs.get("num_points")
        vinit_estimate = np.nanmean(speed[:num_points])
        rinit = np.nanmean(distance_km[:num_points])
        tinit = pd.DatetimeIndex([time[:num_points].mean()])[0]

        inds_rinit = np.where(distance_km >= rinit)[0]
        xdata = speedtime.values[inds_rinit[0]:]
        ydata = distance_km[inds_rinit[0]:]

    elif compute_vinit == 'mean_rsun':
        cutoff_rsun = kwargs.get("cutoff_rsun")
        inds = np.where(distance_rsun <= cutoff_rsun)[0]

        if len(inds) > 0:
            vinit_estimate = np.nanmean(speed[inds])
            rinit = np.nanmean(distance_km[inds])
            tinit = pd.DatetimeIndex([time[inds].mean()])[0]

            inds_rinit = np.where(distance_km >= rinit)[0]
            xdata = speedtime.values[inds_rinit[0]:]
            ydata = distance_km[inds_rinit[0]:]

        else:
            vinit_estimate = speed[0]
            rinit = distance_km[0]
            tinit = time[0]
            xdata = speedtime.values
            ydata = distance_km

    elif compute_vinit == 'mean_donki_cat':
        vinit_input = kwargs.get("vinit_input")

        if vinit_input > 900:
            vinit_donki_cat = 'fast'
        else:
            vinit_donki_cat = 'slow'

        cutoff_rsun = 50 if vinit_donki_cat == 'slow' else 8.4

        inds = np.where(distance_rsun <= cutoff_rsun)[0]

        if len(inds) > 0:
            vinit_estimate = np.nanmean(speed[inds])
            rinit = np.nanmean(distance_km[inds])
            tinit = pd.DatetimeIndex([time[inds].mean()])[0]

            inds_rinit = np.where(distance_km >= rinit)[0]
            xdata = speedtime.values[inds_rinit[0]:]
            ydata = distance_km[inds_rinit[0]:]

        else:
            vinit_estimate = speed[0]
            rinit = distance_km[0]
            tinit = time[0]         

            xdata = speedtime.values
            ydata = distance_km

    fit_vinit = False

    xdata = speedtime.values
    ydata = distance_km
    #testgamma = 1.7863506087704678e-08 
    #testwind  = 200

    #testy = (1/testgamma) * np.log(1 + testgamma*(vinit - testwind) * xdata) + testwind*xdata + rinit

    winds = np.arange(250, 775, 25)
    fit = np.zeros((len(winds), len(xdata)))
    fitspeed = np.zeros((len(winds), len(xdata)))
    residuals = np.zeros((len(winds), len(xdata)))
    gamma = np.zeros(len(winds))
    success = np.zeros(len(winds))
    
    # Initial guess for the parameter gamma
    if fit_vinit:
        initial_guess = [1e-7, vinit_estimate]
        bnds = [(1e-10, max_gamma), (100, 3000)]
    else:
        initial_guess = [1e-7]
        bnds = [(1e-10, max_gamma)]
    #k= 0.3
    
    # do a fit for each wind speed separately and plot valid fits
    for i in range(len(winds)):
        swspeed = winds[i]
        if vinit_estimate > swspeed:
            # Perform the optimisation
            if fit_vinit:
                result = minimize(cost_function_updated, initial_guess,args=(swspeed,rinit,ydata,xdata), method='Nelder-Mead', bounds=bnds)
            else:
                result = minimize(cost_function, initial_guess,args=(vinit_estimate, swspeed,rinit,ydata,xdata), method='Nelder-Mead', bounds=bnds)
            # gamma_fit, pcov = curve_fit(fitdbm, xdata, ydata,p0=initial_guess,method="dogbox")
            # Print the fitted parameter

            success[i] = True
            if result.success:
                gamma_fit = result.x[0]

                if fit_vinit:
                    vinit = result.x[1]
                else:
                    vinit = vinit_estimate

                # fit_ = fitdbm(xdata, gamma_fit)
                fit_ = fitdbm(xdata, gamma_fit, vinit, swspeed, rinit)
                gamma[i] = gamma_fit
                fit[i,:] = fit_   
                residuals[i,:] = np.abs(ydata - fit_)#np.median(np.sqrt((ydata - fit_) ** 2 ))# * logistic_growth(ydata,k)/logistic_growth(ydata,k).max())
                fitspeed[i,:] = np.gradient(fit[i,:], xdata)

            else:
                continue

        else:
            # Perform the optimisation
            if fit_vinit:
                result = minimize(cost_functionneg_updated, initial_guess, args=(swspeed,rinit,ydata,xdata), method='Nelder-Mead', bounds=bnds)
            else:
                result = minimize(cost_functionneg, initial_guess, args=(vinit_estimate, swspeed,rinit,ydata,xdata), method='Nelder-Mead', bounds=bnds)

            success[i] = result.success
            # Print the fitted parameter
            if result.success:
                gamma_fit = result.x[0]

                if fit_vinit:
                    vinit = result.x[1]
                else:
                    vinit = vinit_estimate

                fit_ = fitdbmneg(xdata, gamma_fit, vinit, swspeed, rinit)
                gamma[i] = gamma_fit
                fit[i,:] = fit_   
                residuals[i,:] = np.abs(ydata - fit_)#np.median(np.abs(ydata - fit_)) test because the wrong wind speed is chosen
                fitspeed[i,:] = np.gradient(fit[i,:], xdata)

            else:
                continue
            
    res = np.zeros(len(winds))

    for i in range(len(winds)):
        if success[i]:
            res[i] = np.mean(residuals[i,:])#residuals[i,0] #np.mean(residuals[i,:])
        else:
            res[i] = np.nan
            
    if det_plot:        
        sns.set_context('talk')
        sns.set_style('darkgrid')
        figdbm, ax = plt.subplots(1, 1, figsize = (8,3), dpi = 300, facecolor='white')
        

        ax.set_title('ELEvoHI DBMfits', size = 16)
        ax.set_ylabel('CME Apex Speed [km s$^{-1}$]', fontsize = 14, labelpad = 0)
        ax.set_xlabel('Heliocentric Distance [R$_\odot$]', fontsize = 14, labelpad = 0)
        ax.tick_params(axis = 'both', which = 'major', labelsize = 10, pad = -5)

        norm = mpl.colors.Normalize(vmin = winds.min(), vmax = winds.max())
        cmap = mpl.cm.ScalarMappable(norm = norm, cmap = mpl.cm.jet)
        cmap.set_array([])

        ax = plt.gca()
        cbar = plt.colorbar(cmap, pad = 0.02, ax=ax)
        cbar.set_label('solar wind speed [km s$^{-1}$]', rotation = 270, fontsize = 14, labelpad = 20)
        cbar.ax.tick_params(axis='y', direction='inout', length=5, width=0, labelsize=12)
    
    gamma_v = []
    res_v = []
    winds_v = []
    
        
    residuals2 = []
    for i in range(len(winds)):
        if success[i] != True:
            continue
        if res[i]/rsun > max_residual:
            continue
        if gamma[i] <= 0:
            continue
        if max_gamma >= np.abs(gamma[i]):
            
            gamma_v.append(gamma[i])
            res_v.append(res[i]/rsun)
            winds_v.append(winds[i])

            if det_plot:
                ax.plot(distance_rsun, fitspeed[i,:], label='fit', linewidth=1, c=cmap.to_rgba(winds[i]))
        else:
            continue

    if det_plot:
        ax.plot(distance_rsun, speed, 'o', label='data', c='black', markersize=5)
        
        elcon_out = pd.DataFrame({
                "time": time,
                "R": distance_rsun,
                "V": speed
            })
        
        # elcon_out.to_csv(prediction_path + "ELCon.csv", index=False, header=["time", "heliocentric distance", "speed"])
        
        # Calculate y-axis limits
        # y_lower = min(speed) - 100
        # y_upper = max(speed) + 100
        
        # Use fixed y-axis limits
        y_lower = 0
        y_upper = 1500

        # Set the y-axis limits
        ax.set_ylim(y_lower, y_upper)
    
    # find the most accurate fitting parameters based on the minimum mean residual
    
    if len(gamma_v) != 0:
        gamma_valid = np.array(gamma_v)   
        res_valid = np.array(res_v)   
        winds_valid = np.array(winds_v)   
        min_res = np.argmin(np.abs(res_valid))

    else:
        gamma_valid = [0]
        winds_valid = [0]
        res_valid = [0]
        tinit = 0
        rinit = 0
        vinit = 0
        swspeed = 0
        xdata = [0]
        ydata = [0]
        speed = [0]
        return gamma_valid, winds_valid, res_valid, tinit, rinit, vinit, swspeed, xdata, ydata, speed
    

    # Get the indices that would sort the array based on absolute values
    sorted_indices = np.argsort(np.abs(res_valid))
    gamma_valid = gamma_valid[sorted_indices]
    res_valid = res_valid[sorted_indices]
    winds_valid = winds_valid[sorted_indices]
  
    #pdb.set_trace()
    
    if det_plot:
        filename = prediction_path + '/DBMfit.png' 
        plt.savefig(filename, dpi=300, facecolor=figdbm.get_facecolor(), edgecolor='none', bbox_inches='tight')       
        figdbm.clf()
        plt.close(figdbm)

    
    return gamma_valid, winds_valid, res_valid, tinit, rinit, vinit, swspeed, xdata, ydata, speed

def elevo_analytic(R, f, halfwidth, delta, out=False, plot=False):
    """
    Calculate the distance from the Sun to a point along an ellipse in the ecliptic plane, which 
    describes the front of a solar transient. This point is specified by the angle delta 
    between the ellipse apex and the in situ spacecraft. The ellipse is specified by 
    the distance of the apex (R), the half width in degrees in heliospheric longitude, and the aspect ratio.
    One of the ellipse main axes is oriented along the propagation direction, the 
    other perpendicular to it. The aspect ratio can take any values, but physical 
    ones suitable for CMEs cluster most likely around 1.3 +/ 0.2.

    Authors: Christian Möstl, Tanja Amerstorfer, Jürgen Hinterreiter, Maike Bauer
    
    Parameters: 
        R (numpy.ndarray): Heliocentric distance of CME apex in AU
        aspectratio (float): Ellipse aspect ratio (a/b)
        halfwidth (float): Halfwith of the ellipse
        delta (float): Angular separation of CME apex and in situ s/c

    Returns:
        numpy.ndarray: Heliocentric distance of the CME front along the Sun-s/c line.

    History:
        2014/09: Version 1.0 numerical solution (C. Möstl)
        2014/10: Replaced numerical with analytic procedure (C. Möstl)
        2015/11 - 2019/08: several changes related to ELEvoHI (T. Amerstorfer, J. Hinterreiter)
        2023/08: translated from IDL to Python (T. Amerstorfer)

    """
    
    aspectratio = 1/f


    # Construct ellipse
    theta = np.arctan(f**2 * np.tan(halfwidth))
    omega = np.sqrt(np.cos(theta)**2 * (f**2 - 1) + 1)
    b = R * omega * np.sin(halfwidth) / (np.cos(halfwidth - theta) + omega * np.sin(halfwidth))
    a = b / f
    c = R - b

    # Ellipse parameters necessary for plotting
    
    if plot:
        return a, b, c
    else:
        # Distance from Sun of given point along ellipse front
        root = np.sqrt(np.sin(delta)**2 * f**2 * (b**2 - c**2) + np.cos(delta)**2 * b**2)
        dvalue_analytic_front = (c * np.cos(delta) + root) / (np.sin(delta)**2 * f**2 + np.cos(delta)**2)
        #dvalue_analytic_rear = (c * np.cos(delta) - root) / (np.sin(delta)**2 * f**2 + np.cos(delta)**2)
        dvalue = dvalue_analytic_front
        
        return dvalue

def elevo_analytic_new(R, f, halfwidth, delta):
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

def elevo(R, time_array, tnum, direction, f, halfwidth, vdrag, track, availability, hit_counts, delta_values, positions, HIobs, outer_system, prediction_path, det_plot,  movie=False, timegrid = 1440):
    # calculate ELEvo radial distances in direction of each target and
    # arrival times and speeds
    
    # Assign values from dictionary to variables
    sta_available = availability['sta_available']
    stb_available = availability['stb_available']
    mes_available = availability['mes_available']
    vex_available = availability['vex_available']
    solo_available = availability['solo_available']
    psp_available = availability['psp_available']
    bepi_available = availability['bepi_available']
    
    hit_sta = hit_counts['hit_sta']
    hit_stb = hit_counts['hit_stb']
    hit_mes = hit_counts['hit_mes']    
    hit_vex = hit_counts['hit_vex']
    hit_solo = hit_counts['hit_solo']
    hit_psp = hit_counts['hit_psp']
    hit_bepi = hit_counts['hit_bepi']
    hit_L1 = hit_counts['hit_L1']
    hit_mercury = hit_counts['hit_mercury']
    hit_venus = hit_counts['hit_venus']
    hit_earth = hit_counts['hit_earth']
    hit_mars = hit_counts['hit_mars']
    
    if outer_system:
        hit_jupiter = hit_counts['hit_jupiter']
        hit_saturn = hit_counts['hit_saturn']
        hit_uranus = hit_counts['hit_uranus']
        hit_neptune = hit_counts['hit_neptune']
        delta_jupiter = delta_values['delta_jupiter']
        delta_saturn = delta_values['delta_saturn']
        delta_uranus = delta_values['delta_uranus']
        delta_neptune = delta_values['delta_neptune']
    
    if sta_available:
        delta_sta = delta_values['delta_sta']
    if stb_available:
        delta_stb = delta_values['delta_stb']
    if mes_available:
        delta_mes = delta_values['delta_mes']
    if vex_available:
        delta_vex = delta_values['delta_vex']
    if psp_available:
        delta_psp = delta_values['delta_psp']
    if solo_available:
        delta_solo = delta_values['delta_solo']
    if bepi_available:
        delta_bepi = delta_values['delta_bepi']
        
    delta_L1 = delta_values['delta_L1']
    delta_mercury = delta_values['delta_mercury']
    delta_venus = delta_values['delta_venus']
    delta_earth = delta_values['delta_earth']
    delta_mars = delta_values['delta_mars']
      
    mars_r = positions['mars_r']
    earth_r = positions['earth_r']
    venus_r = positions['venus_r']
    mercury_r = positions['mercury_r']
    L1_r = positions['L1_r']
    mars_lon = positions['mars_lon']
    earth_lon = positions['earth_lon']
    venus_lon = positions['venus_lon']
    mercury_lon = positions['mercury_lon']
    L1_lon = positions['L1_lon']
    
    if sta_available:
        sta_r = positions['sta_r']
        sta_lon = positions['sta_lon']        
    if stb_available:
        stb_r = positions['stb_r']
        stb_lon = positions['stb_lon']    
    if bepi_available:   
        bepi_r = positions['bepi_r']
        bepi_lon = positions['bepi_lon']    
    if solo_available:
        solo_r = positions['solo_r']
        solo_lon = positions['solo_lon']
    if psp_available:
        psp_r = positions['psp_r']
        psp_lon = positions['psp_lon']
    if vex_available:
        vex_r = positions['vex_r']
        vex_lon = positions['vex_lon']
    if mes_available:
        mes_r = positions['mes_r']
        mes_lon = positions['mes_lon']
    if outer_system:
        neptune_r = positions['neptune_r']
        uranus_r = positions['uranus_r']
        saturn_r = positions['saturn_r']
        jupiter_r = positions['jupiter_r']
        neptune_lon = positions['neptune_lon']
        uranus_lon = positions['uranus_lon']
        saturn_lon = positions['saturn_lon']
        jupiter_lon = positions['jupiter_lon']
          
    pred = []
    
    # Calculate time differences needed for deriving the speeds
    time_diff = np.diff(tnum)
    
    hit = 0    

    if sta_available == 1 and hit_sta == 1:
        ############# STEREO-A #############
        d_sta = elevo_analytic(R, f, halfwidth, delta_sta, out=False)

        # Find the index of the closest value
        index_sta = np.argmin(np.abs(d_sta - sta_r))

        # arrival time at STEREO-A
        arrtime_sta = time_array[index_sta]

        # arrival speed at STEREO-A
        distance_diff = np.diff(d_sta*AU)
        speed_sta = distance_diff / time_diff
        arrspeed_sta = speed_sta[index_sta-1] # first element is cut off during building the time derivative

        hit = 1
        
        pred.append({"target": "STEREO-A", "arrival time [UT]": arrtime_sta.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_sta)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if stb_available == 1 and hit_stb == 1:
        ############# STEREO-B #############
        d_stb = elevo_analytic(R, f, halfwidth, delta_stb, out=False)

        # Find the index of the closest value
        index_stb = np.argmin(np.abs(d_stb - stb_r))

        # arrival time at STEREO-B
        arrtime_stb = time_array[index_stb]

        # arrival speed at STEREO-B
        distance_diff = np.diff(d_stb*AU)
        speed_stb = distance_diff / time_diff
        arrspeed_stb = speed_stb[index_stb-1] # first element is cut off during building the time derivative
        
        hit = 1

        pred.append({"target": "STEREO-B", "arrival time [UT]": arrtime_stb.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_stb)), "dt [h]": np.nan, "dv [km/s]": np.nan})        
        
    if mes_available == 1 and hit_mes == 1:
        ############# MESSENGER #############
        d_mes = elevo_analytic(R, f, halfwidth, delta_mes, out=False)

        # Find the index of the closest value
        index_mes = np.argmin(np.abs(d_mes - mes_r))

        # arrival time at MESSENGER
        arrtime_mes = time_array[index_mes]

        # arrival speed at MESSENGER
        distance_diff = np.diff(d_mes*AU)
        speed_mes = distance_diff / time_diff
        arrspeed_mes = speed_mes[index_mes-1] # first element is cut off during building the time derivative
        
        hit = 1

        pred.append({"target": "MESSENGER", "arrival time [UT]": arrtime_mes.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_mes)), "dt [h]": np.nan, "dv [km/s]": np.nan})
        
    if vex_available == 1 and hit_vex == 1:
        ############# Venus Express #############
        d_vex = elevo_analytic(R, f, halfwidth, delta_vex, out=False)

        # Find the index of the closest value
        index_vex = np.argmin(np.abs(d_vex - vex_r))

        # arrival time at Venus Express
        arrtime_vex = time_array[index_vex]

        # arrival speed at Venus Express
        distance_diff = np.diff(d_vex*AU)
        speed_vex = distance_diff / time_diff
        arrspeed_vex = speed_vex[index_vex-1] # first element is cut off during building the time derivative
        
        hit = 1
        
        pred.append({"target": "Venus Express", "arrival time [UT]": arrtime_vex.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_vex)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if solo_available == 1 and hit_solo == 1:
        ############# Solar Orbiter #############
        d_solo = elevo_analytic(R, f, halfwidth, delta_solo, out=False)

        # Find the index of the closest value
        index_solo = np.argmin(np.abs(d_solo - solo_r))

        # arrival time at Solar Orbiter
        arrtime_solo = time_array[index_solo]

        # arrival speed at Solar Orbiter
        distance_diff = np.diff(d_solo*AU)
        speed_solo = distance_diff / time_diff
        arrspeed_solo = speed_solo[index_solo-1] # first element is cut off during building the time derivative
        
        hit = 1
        
        pred.append({"target": "Solar Orbiter", "arrival time [UT]": arrtime_solo.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_solo)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if psp_available == 1 and hit_psp == 1:
        ############# Parker Solar Probe #############
        d_psp = elevo_analytic(R, f, halfwidth, delta_psp, out=False)

        # Find the index of the closest value
        index_psp = np.argmin(np.abs(d_psp - psp_r))

        # arrival time at Parker Solar Probe
        arrtime_psp = time_array[index_psp]

        # arrival speed at Parker Solar Probe
        distance_diff = np.diff(d_psp*AU)
        speed_psp = distance_diff / time_diff
        arrspeed_psp = speed_psp[index_psp-1] # first element is cut off during building the time derivative
        
        hit = 1
        
        pred.append({"target": "Parker Solar Probe", "arrival time [UT]": arrtime_psp.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_psp)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if bepi_available == 1 and hit_bepi == 1:
        ############# BepiColombo #############
        d_bepi = elevo_analytic(R, f, halfwidth, delta_bepi, out=False)

        # Find the index of the closest value
        index_bepi = np.argmin(np.abs(d_bepi - bepi_r))

        # arrival time at BepiColombo
        arrtime_bepi = time_array[index_bepi]

        # arrival speed at BepiColombo
        distance_diff = np.diff(d_bepi*AU)
        speed_bepi = distance_diff / time_diff
        arrspeed_bepi = speed_bepi[index_bepi-1] # first element is cut off during building the time derivative
        
        hit = 1
        
        pred.append({"target": "BepiColombo", "arrival time [UT]": arrtime_bepi.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_bepi)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if hit_mercury == 1:
        ############# Mercury #############
        d_mercury = elevo_analytic(R, f, halfwidth, delta_mercury,  out=False)

        # Find the index of the closest value
        index_mercury = np.argmin(np.abs(d_mercury - mercury_r))

        # arrival time at Mercury
        arrtime_mercury = time_array[index_mercury]

        # arrival speed at Mercury
        distance_diff = np.diff(d_mercury*AU)
        speed_mercury = distance_diff / time_diff
        arrspeed_mercury = speed_mercury[index_mercury-1] # first element is cut off during building the time derivative
        
        hit = 1
        
        pred.append({"target": "Mercury", "arrival time [UT]": arrtime_mercury.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_mercury)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if hit_venus == 1:
        ############# Venus #############
        d_venus = elevo_analytic(R, f, halfwidth, delta_venus,  out=False)

        # Find the index of the closest value
        index_venus = np.argmin(np.abs(d_venus - venus_r))

        # arrival time at Venus
        arrtime_venus = time_array[index_venus]

        # arrival speed at Venus
        distance_diff = np.diff(d_venus*AU)
        speed_venus = distance_diff / time_diff
        arrspeed_venus = speed_venus[index_venus-1] # first element is cut off during building the time derivative
        
        hit = 1
        
        pred.append({"target": "Venus", "arrival time [UT]": arrtime_venus.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_venus)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if hit_L1 == 1:
        ############# L1 #############
        d_L1 = elevo_analytic(R, f, halfwidth, delta_L1,  out=False)

        # Find the index of the closest value
        index_L1 = np.argmin(np.abs(d_L1 - L1_r))

        # arrival time at L1
        arrtime_L1 = time_array[index_L1]

        # arrival speed at L1
        distance_diff = np.diff(d_L1*AU)
        speed_L1 = distance_diff / time_diff
        arrspeed_L1 = speed_L1[index_L1-1] # first element is cut off during building the time derivative
        
        hit = 1
        
        pred.append({"target": "L1", "arrival time [UT]": arrtime_L1.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_L1)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if hit_mars == 1:
        ############# Mars #############
        d_mars = elevo_analytic(R, f, halfwidth, delta_mars,  out=False)

        # Find the index of the closest value
        index_mars = np.argmin(np.abs(d_mars - mars_r))

        # arrival time at Mars
        arrtime_mars = time_array[index_mars]

        # arrival speed at Mars
        distance_diff = np.diff(d_mars*AU)
        speed_mars = distance_diff / time_diff
        arrspeed_mars = speed_mars[index_mars-1] # first element is cut off during building the time derivative
        
        hit = 1
        
        pred.append({"target": "Mars", "arrival time [UT]": arrtime_mars.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_mars)), "dt [h]": np.nan, "dv [km/s]": np.nan})

    if outer_system == 1:

        if hit_jupiter == 1:
            ############# Jupiter #############
            d_jupiter = elevo_analytic(R, f, halfwidth, delta_jupiter,  out=False)

            # Find the index of the closest value
            index_jupiter = np.argmin(np.abs(d_jupiter - jupiter_r))

            # arrival time at Jupiter
            arrtime_jupiter = time_array[index_jupiter]

            # arrival speed at Jupiter
            distance_diff = np.diff(d_jupiter*AU)
            speed_jupiter = distance_diff / time_diff
            arrspeed_jupiter = speed_jupiter[index_jupiter-1] # first element is cut off during building the time derivative
            
            hit = 1
            
            pred.append({"target": "Jupiter", "arrival time [UT]": arrtime_jupiter.strftime("%Y-%m-%d %H:%M"),
                         "arrival speed [km/s]": int(round(arrspeed_jupiter)), "dt [h]": np.nan, "dv [km/s]": np.nan})

        if hit_saturn == 1:
            ############# Saturn #############
            d_saturn = elevo_analytic(R, f, halfwidth, delta_saturn,  out=False)

            # Find the index of the closest value
            index_saturn = np.argmin(np.abs(d_saturn - saturn_r))

            # arrival time at Saturn
            arrtime_saturn = time_array[index_saturn]

            # arrival speed at Saturn
            distance_diff = np.diff(d_saturn*AU)
            speed_saturn = distance_diff / time_diff
            arrspeed_saturn = speed_saturn[index_saturn-1] # first element is cut off during building the time derivative
            
            hit = 1
            
            pred.append({"target": "Saturn", "arrival time [UT]": arrtime_saturn.strftime("%Y-%m-%d %H:%M"),
                         "arrival speed [km/s]": int(round(arrspeed_saturn)), "dt [h]": np.nan, "dv [km/s]": np.nan})

        if hit_uranus == 1:
            ############# Uranus #############
            d_uranus = elevo_analytic(R, f, halfwidth, delta_uranus,  out=False)

            # Find the index of the closest value
            index_uranus = np.argmin(np.abs(d_uranus - uranus_r))

            # arrival time at Uranus
            arrtime_uranus = time_array[index_uranus]

            # arrival speed at Uranus
            distance_diff = np.diff(d_mars*AU)
            speed_uranus = distance_diff / time_diff
            arrspeed_uranus = speed_uranus[index_uranus-1] # first element is cut off during building the time derivative

            hit = 1
            
            pred.append({"target": "Uranus", "arrival time [UT]": arrtime_uranus.strftime("%Y-%m-%d %H:%M"),
                         "arrival speed [km/s]": int(round(arrspeed_uranus)), "dt [h]": np.nan, "dv [km/s]": np.nan})

        if hit_neptune == 1:
            ############# Neptune #############
            d_neptune = elevo_analytic(R, f, halfwidth, delta_neptune,  out=False)

            # Find the index of the closest value
            index_neptune = np.argmin(np.abs(d_neptune - neptune_r))

            # arrival time at Neptune
            arrtime_neptune = time_array[index_neptune]

            # arrival speed at Neptune
            distance_diff = np.diff(d_neptune*AU)
            speed_neptune = distance_diff / time_diff
            arrspeed_neptune = speed_neptune[index_neptune-1] # first element is cut off during building the time derivative
            
            hit = 1
            
            pred.append({"target": "Neptune", "arrival time [UT]": arrtime_neptune.strftime("%Y-%m-%d %H:%M"),
                         "arrival speed [km/s]": int(round(arrspeed_neptune)), "dt [h]": np.nan, "dv [km/s]": np.nan})
    
    if not hit:
        pred.append({"target": np.nan, "arrival time [UT]": np.nan,
                         "arrival speed [km/s]": np.nan, "dt [h]": np.nan, "dv [km/s]": np.nan})
    
    if det_plot:
        # Create a figure with two subplots (panels)
        sns.set_context('talk')
        sns.set_style('darkgrid')
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), facecolor='white')

        # Heliocentric distance
        axs[0].plot(time_array, R, label='CME apex')
        if hit_L1:
            axs[0].plot(time_array, d_L1, label='L1')
        
        axs[0].set_ylabel("Heliocentric Distance [AU]")
        axs[0].set_title("ELEvo kinematics - Heliocentric Distance")
        axs[0].grid(True)
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axs[0].legend()

        # CME speed
        axs[1].plot(time_array[1:], vdrag[1:], label='CME apex')
        if hit_L1:
            axs[1].plot(time_array[1:], speed_L1, label='L1')
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("CME Speed [km s$^{-1}$]")
        axs[1].set_title("ELEvo kinematics - CME Speed")
        axs[1].grid(True)
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axs[1].xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Adjust locator for better date spacing
        axs[1].legend()

        # Calculate y-axis limits
        y_lower = min(vdrag) - 100
        y_upper = max(vdrag) + 100

        # Set the y-axis limits
        axs[1].set_ylim(y_lower, y_upper)

        # Rotate x-axis labels for both panels
        for ax in axs:
            ax.xaxis.set_tick_params(rotation=45)

        plt.tight_layout()

        filename = prediction_path + '/ELEvo.png' 

        plt.savefig(filename, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        
        fig.clf()
        plt.close(fig)
    
    prediction = pd.DataFrame(pred)
    
    # interpolate time-elongation track to movie time-axis to display the tangent to the CME front
    # Convert datetime values to Unix timestamps in seconds
    elon = track["elongation"]

    # existing time axis from tracking
    track["time"] = pd.to_datetime(track["time"])
    x = (track["time"] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    
    # Create an interpolation function using interp1d
    interp_func = interp1d(x, elon, kind='linear', fill_value="extrapolate")

    # Convert new_time_axis to Unix timestamps in seconds
    time_a = pd.to_datetime(time_array)
    
    # Convert datetime values to Unix timestamps in seconds
    x_movie = (time_a - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    
    x_last_element = x.iloc[-1]  # Get the last element of track time

    # Check which elements in x_new are greater than or equal to the last element of x
    smaller_than_last_x = np.where(x_movie <= x_last_element)[0]
    
    x_new = x_movie[smaller_than_last_x]

    # Interpolate elongation onto the common new time axis
    # This new time axis ends with endcut defined in config file
    interp_elon = interp_func(x_new)
    
    # Define the starting point of the tangent
    if HIobs == 'A':
        start_angle = sta_lon
        start_radius = sta_r
    if HIobs == 'B':
        start_angle = stb_lon
        start_radius = stb_r

    # Define the length of the tangent
    tangent_length = 1.2
    
    # Define the length of the HI fov shading
    fov_length = 1
    
    elon_rad = np.deg2rad(interp_elon)
    #pdb.set_trace()
   
    if det_plot:
        
        sns.set_context('talk')
        sns.set_style('darkgrid')
        
        # Initialize your plot outside of the loop
        if movie:
            if os.path.isdir(prediction_path + '/frames') == False: os.mkdir(prediction_path + '/frames')

        # Loop over time frames
    
        for k in range(timegrid):
            
            plt.close('all')
            
            figmov = plt.figure(figsize=(10, 10))
            backcolor = 'black'
            
            t = (np.arange(181) * np.pi/180) - direction
            t1 = (np.arange(181) * np.pi/180)

            a, b, c = elevo_analytic(R[k]*AU, f, halfwidth, 0.,  out=False, plot=True)
            a = a/AU
            b = b/AU
            c = c/AU

            xc = c * np.cos(direction) + ((a * b) / np.sqrt((b * np.cos(t1)) ** 2 + (a * np.sin(t1)) ** 2)) * np.sin(t)
            yc = c * np.sin(direction) + ((a * b) / np.sqrt((b * np.cos(t1)) ** 2 + (a * np.sin(t1)) ** 2)) * np.cos(t)

            theta_ell = np.arctan2(yc, xc)
            r_ell = np.sqrt(xc ** 2 + yc ** 2)

            if np.min(r_ell) > 1.2:
                break

            # Create a subplot for the current frame
            ax = figmov.add_subplot(projection='polar')

            # Set the title according to the time step
            ax.set_title(time_array[k].strftime('%Y-%m-%d %H:%M'))

            ax.plot(theta_ell, r_ell, color="tab:orange")

            # plot the position of the planets
            ax.scatter(mercury_lon, mercury_r, color = 'dimgrey', marker = 'o', label = 'Point')
            ax.scatter(venus_lon, venus_r, color = 'orange', marker = 'o', label = 'Point')
            ax.scatter(earth_lon, earth_r, color = 'mediumseagreen', marker = 'o', label = 'Point')
            ax.scatter(mars_lon, mars_r, color = 'orangered', marker = 'o', label = 'Point')

            # plot the position of the s/c
            if sta_available:
                ax.scatter(sta_lon, sta_r, color = 'red', marker = 's', label = 'Point')
            if stb_available:
                ax.scatter(stb_lon, stb_r, color = 'blue', marker = 's', label = 'Point')
            if mes_available:
                ax.scatter(mes_lon, mes_r, color = 'dimgrey', marker = 's', label = 'Point')
            if vex_available:
                ax.scatter(vex_lon, vex_r, color = 'orange', marker = 's', label = 'Point')
            if psp_available:
                ax.scatter(psp_lon, psp_r, color = 'black', marker = 's', label = 'Point')
            if solo_available:
                ax.scatter(solo_lon, solo_r, color = 'coral', marker = 's', label = 'Point')
            if bepi_available:
                ax.scatter(bepi_lon, bepi_r, color = 'blue', marker = 's', label = 'Point')

            if outer_system == 0:
                ax.set_rgrids((0.2, 0.4, 0.6, 0.8, 1.0), ('0.2', '0.4', '0.6', '0.8', '1 AU'), angle = 125, fontsize = 12,
                            alpha = 0.5, color = backcolor)
                ax.set_ylim(0, 1.2) 
            else:
                ax.scatter(jupiter_lon, jupiter_r, color = 'darkgoldenrod', marker = 'o', label = 'Point')
                ax.scatter(saturn_lon, saturn_r, color = 'palegreen', marker = 'o', label = 'Point')
                ax.scatter(uranus_lon, uranus_r, color = 'cyan', marker = 'o', label = 'Point')
                ax.scatter(neptune_lon, neptune_r, color = 'cornflowerblue', marker = 'o', label = 'Point')

                plt.rgrids((5, 10, 15, 20, 25, 30), ('5', '10', '15', '20', '25', '30 AU'), angle = 125, fontsize = 12,
                            alpha = 0.5, color = backcolor)
                ax.set_ylim(0, 32)
                
            #######################################################
            # shade HI field of view
            # HI1 field of view:
            hi1_fov_start = np.deg2rad(4.)
            hi1_fov_end = np.deg2rad(24.)
            
            
#---- inner HI1 fov ------
            if HIobs == 'A':
                end_hi1_innerradius = np.sqrt(fov_length**2 + sta_r**2 - 2. * fov_length * sta_r * np.cos(hi1_fov_start))
                
                if sta_lon < 0:
                    end_hi1_innerangle = abs(np.arcsin((fov_length * np.sin(hi1_fov_start)) / end_hi1_innerradius) - np.pi) - abs(start_angle)                
                else:
                    end_hi1_innerangle = (abs(np.arcsin((fov_length * np.sin(hi1_fov_start)) / end_hi1_innerradius) - np.pi) - abs(start_angle)) * (-1)        
                
                line_x_hi1_inner = np.array([start_angle, end_hi1_innerangle])
                line_y_hi1_inner = np.array([start_radius, end_hi1_innerradius])
                #---- outer HI1 fov ------
                end_hi1_outerradius = np.sqrt(fov_length**2 + sta_r**2 - 2. * fov_length * sta_r * np.cos(hi1_fov_end))
                
                if sta_lon < 0:
                    end_hi1_outerangle = abs(np.arcsin((fov_length * np.sin(hi1_fov_end)) / end_hi1_outerradius)) - abs(start_angle)                
                else:
                    end_hi1_outerangle = (abs(np.arcsin((fov_length * np.sin(hi1_fov_end)) / end_hi1_outerradius)) - abs(start_angle)) * (-1)         
                    
                line_x_hi1_outer = np.array([start_angle, end_hi1_outerangle])
                line_y_hi1_outer = np.array([start_radius, end_hi1_outerradius])

                # HI2 field of view:
                hi2_fov_start = np.deg2rad(18.)
                hi2_fov_end = np.deg2rad(88.)

                #---- inner HI2 fov ------
                end_hi2_innerradius = np.sqrt(fov_length**2 + sta_r**2 - 2. * fov_length * sta_r * np.cos(hi2_fov_start))
                
                if sta_lon < 0:
                    end_hi2_innerangle = abs(np.arcsin((fov_length * np.sin(hi2_fov_start)) / end_hi2_innerradius) - np.pi) - abs(start_angle)                
                else:
                    end_hi2_innerangle = (abs(np.arcsin((fov_length * np.sin(hi2_fov_start)) / end_hi2_innerradius) - np.pi) - abs(start_angle)) * (-1)               
                
                line_x_hi2_inner = np.array([start_angle, end_hi2_innerangle])
                line_y_hi2_inner = np.array([start_radius, end_hi2_innerradius])
                
                #---- outer HI2 fov ------
                end_hi2_outerradius = np.sqrt(fov_length**2 + sta_r**2 - 2. * fov_length * sta_r * np.cos(hi2_fov_end))
            
                if sta_lon < 0:
                    end_hi2_outerangle = abs(np.arcsin((fov_length * np.sin(hi2_fov_end)) / end_hi2_outerradius)) - abs(start_angle)                
                else:
                    end_hi2_outerangle = (abs(np.arcsin((fov_length * np.sin(hi2_fov_end)) / end_hi2_outerradius)) - abs(start_angle)) * (-1)

            if HIobs == 'B':
                end_hi1_innerradius = np.sqrt(fov_length**2 + stb_r**2 - 2. * fov_length * stb_r * np.cos(hi1_fov_start))
                
                if stb_lon < 0:
                    end_hi1_innerangle = abs(np.arcsin((fov_length * np.sin(hi1_fov_start)) / end_hi1_innerradius) - np.pi) - abs(start_angle)                
                else:
                    end_hi1_innerangle = (abs(np.arcsin((fov_length * np.sin(hi1_fov_start)) / end_hi1_innerradius) - np.pi) - abs(start_angle)) * (-1)        
                
                line_x_hi1_inner = np.array([start_angle, end_hi1_innerangle])
                line_y_hi1_inner = np.array([start_radius, end_hi1_innerradius])
                #---- outer HI1 fov ------
                end_hi1_outerradius = np.sqrt(fov_length**2 + stb_r**2 - 2. * fov_length * stb_r * np.cos(hi1_fov_end))
                
                if stb_lon < 0:
                    end_hi1_outerangle = abs(np.arcsin((fov_length * np.sin(hi1_fov_end)) / end_hi1_outerradius)) - abs(start_angle)                
                else:
                    end_hi1_outerangle = (abs(np.arcsin((fov_length * np.sin(hi1_fov_end)) / end_hi1_outerradius)) - abs(start_angle)) * (-1)         
                    
                line_x_hi1_outer = np.array([start_angle, end_hi1_outerangle])
                line_y_hi1_outer = np.array([start_radius, end_hi1_outerradius])

                # HI2 field of view:
                hi2_fov_start = np.deg2rad(18.)
                hi2_fov_end = np.deg2rad(88.)

                #---- inner HI2 fov ------
                end_hi2_innerradius = np.sqrt(fov_length**2 + stb_r**2 - 2. * fov_length * stb_r * np.cos(hi2_fov_start))
                
                if stb_lon < 0:
                    end_hi2_innerangle = abs(np.arcsin((fov_length * np.sin(hi2_fov_start)) / end_hi2_innerradius) - np.pi) - abs(start_angle)                
                else:
                    end_hi2_innerangle = (abs(np.arcsin((fov_length * np.sin(hi2_fov_start)) / end_hi2_innerradius) - np.pi) - abs(start_angle)) * (-1)               
                
                line_x_hi2_inner = np.array([start_angle, end_hi2_innerangle])
                line_y_hi2_inner = np.array([start_radius, end_hi2_innerradius])
                
                #---- outer HI2 fov ------
                end_hi2_outerradius = np.sqrt(fov_length**2 + stb_r**2 - 2. * fov_length * stb_r * np.cos(hi2_fov_end))
            
                if stb_lon < 0:
                    end_hi2_outerangle = abs(np.arcsin((fov_length * np.sin(hi2_fov_end)) / end_hi2_outerradius)) - abs(start_angle)                
                else:
                    end_hi2_outerangle = (abs(np.arcsin((fov_length * np.sin(hi2_fov_end)) / end_hi2_outerradius)) - abs(start_angle)) * (-1)
                            
            line_x_hi2_outer = np.array([start_angle, end_hi2_outerangle])
            line_y_hi2_outer = np.array([start_radius, end_hi2_outerradius])

            # Shade the region between the two lines
            ax.plot(line_x_hi1_inner, line_y_hi1_inner, color='gray', linestyle='-', linewidth=2, alpha=0.1)
            ax.plot(line_x_hi1_outer, line_y_hi1_outer, color='gray', linestyle='-', linewidth=2, alpha=0.1)

            ax.plot(line_x_hi2_inner, line_y_hi2_inner, color='gray', linestyle='-', linewidth=2, alpha=0.1)
            ax.plot(line_x_hi2_outer, line_y_hi2_outer, color='gray', linestyle='-', linewidth=2, alpha=0.1)

            if k < len(elon_rad)-1:
            ######
            # Calculate the ending point of the tangent
                if HIobs == 'A':

                    end_radius = np.sqrt(tangent_length**2 + sta_r**2 - 2. * tangent_length * sta_r * np.cos(elon_rad[k]))

                    # angle of end point of tangent
                    if np.cos(elon_rad[k]) > (sta_r/tangent_length): 
                        beta = np.arcsin((tangent_length * np.sin(elon_rad[k])) / end_radius) - np.pi
                    else:
                        beta = np.arcsin((tangent_length * np.sin(elon_rad[k])) / end_radius)

                    if sta_lon < 0:
                        end_angle = abs(beta) - abs(start_angle)
                    else:
                        end_angle = (abs(beta) - abs(start_angle)) * (-1)
                        
                if HIobs == 'B':

                    end_radius = np.sqrt(tangent_length**2 + stb_r**2 - 2. * tangent_length * stb_r * np.cos(elon_rad[k]))

                    # angle of end point of tangent
                    if np.cos(elon_rad[k]) > (stb_r/tangent_length): 
                        beta = np.arcsin((tangent_length * np.sin(elon_rad[k])) / end_radius) - np.pi
                    else:
                        beta = np.arcsin((tangent_length * np.sin(elon_rad[k])) / end_radius)

                    if stb_lon < 0:
                        end_angle = abs(beta) - abs(start_angle)
                    else:
                        end_angle = (abs(beta) - abs(start_angle)) * (-1)

                # Calculate the coordinates of the line
                line_x = np.array([start_angle, end_angle])
                line_y = np.array([start_radius, end_radius])
                
                # Plot the HI tangent
                ax.plot(line_x, line_y, color='red', linestyle='-', linewidth=2)
                
            if k==np.round((0.5*len(elon_rad)-1)):
            #save single figure
                filename = prediction_path + 'ELEvoHI_HEE_m.jpg'
                plt.savefig(filename, dpi=300, facecolor=figmov.get_facecolor(), edgecolor='none', bbox_inches='tight')
                if not movie:
                    plt.close(figmov)
                    break
    
            if movie:                
            #save frames
                framestr = '%05i' % (k)
                filename = prediction_path + '/frames/frame_' + framestr + '.jpg' 
                plt.savefig(filename, dpi=300, facecolor=figmov.get_facecolor(), edgecolor='none', bbox_inches='tight')
                plt.close(figmov)
        
        if movie:
            os.system(
                f'ffmpeg -framerate 60 -i "{prediction_path}/frames/frame_%05d.jpg" '
                f'-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" '
                f'-c:v libx264 -pix_fmt yuv420p -b:v 5000k -movflags +faststart '
                f'"{prediction_path}/movie.mp4" -y'
            )
        
            plt.close('all')
    
   
    return prediction

def elevo_new(R, time_array, tnum, f, halfwidth, hit, delta_values, L1_r,L1_lon):
    # calculate ELEvo radial distances in direction of each target and
    # arrival times and speeds
    
   
    hit_L1 = hit
        
    delta_L1 = delta_values
       
    pred = []
    
    # Calculate time differences needed for deriving the speeds
    time_diff = np.diff(tnum)
    

    if hit_L1 == 1:

        ############# L1 #############
        d_L1 = elevo_analytic(R, f, halfwidth, delta_L1, out=False)

        # Find the index of the closest value
        index_L1 = np.argmin(np.abs(d_L1 - L1_r))

        # arrival time at L1
        arrtime_L1 = time_array[index_L1]

        # arrival speed at L1
        distance_diff = np.diff(d_L1*AU)
        speed_L1 = distance_diff / time_diff
        arrspeed_L1 = speed_L1[index_L1-1] # first element is cut off during building the time derivative
        
        hit = 1
        return {"target": "L1", "arrival time [UT]": arrtime_L1.strftime("%Y-%m-%d %H:%M"),
                     "arrival speed [km/s]": int(round(arrspeed_L1)), "dt [h]": np.nan, "dv [km/s]": np.nan}, d_L1
   

    return None, None

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
        f"arr_time_fin": arr_time_mean,
        f"arr_time_err0": arr_time_err_lower,
        f"arr_time_err1": arr_time_err_upper,
        f"arr_time_err_list": err_arr_time,
        f"arr_hit": [1.0],
        f"arr_speed_list": arr_speed_mean,
        f"arr_speed_err_list": err_arr_speed,
    }

def compute_arrival_wrapper(R, vdrag, time_array, L1_r):

    computed_arrival = compute_arrival(R, vdrag, time_array, L1_r)

    pred = {"target": "L1",
            "arrival time [UT]": computed_arrival['arr_time_fin'][0].replace(second=0, microsecond=0),
            "arrival speed [km/s]": int(round(computed_arrival['arr_speed_list'][0])),
            "dt [h]": np.nan,
            "dv [km/s]": np.nan}
    
    return pred, computed_arrival

def assess_prediction(prediction, target, is_time, is_speed):
    
    lead_time = np.nan
# difference of prediction and in situ arrival
# Find the row where "target" is "L1" and access the "arrival time" 
    if isinstance(is_time, datetime):
        arrtime__target = prediction['arrival time [UT]']

        # Convert the "arrival time" to a datetime object
        arrtime_target = pd.to_datetime(arrtime__target)
        arrival_dt = (arrtime_target - is_time).total_seconds()/3600.
        
        prediction['dt [h]'] = round(arrival_dt, 2)
        
        formatted_hours = "{:.2f}".format(arrival_dt)

        if isinstance(is_speed, int):
            arrspeed_target = prediction['arrival speed [km/s]']        
            arrival_dv = arrspeed_target - is_speed
            
            prediction['dv [km/s]'] = int(round(arrival_dv))

        else:
            arrival_dv = np.nan
    else:
            arrival_dt = np.nan
            arrival_dv = np.nan
    
    if not np.isnan(arrival_dt):
        arrival_dt = round(arrival_dt,2)
    if not np.isnan(arrival_dv):
        arrival_dv = int(round(arrival_dv))

    return arrival_dt, arrival_dv, prediction

def assess_ensemble(ensemble, det_results, det_run_no, no_det_run):
    
    ensemble_results = pd.DataFrame()
    
    target_names = ensemble['target'].unique()   
    number_of_runs = ensemble['run no.'].max() 
    
    j = 0
    
    for i in range(0, len(target_names)):
        if target_names[i] == 'No hit!':
            continue
                
        ensemble['arrival time [UT]'] = pd.to_datetime(ensemble['arrival time [UT]'])
        
        mean_arrival_time = ensemble[ensemble['target'] == target_names[i]]['arrival time [UT]'].mean()
        mean_arrival_time = mean_arrival_time.strftime("%Y-%m-%d %H:%M")
        median_arrival_time = ensemble[ensemble['target'] == target_names[i]]['arrival time [UT]'].median()
        median_arrival_time = median_arrival_time.strftime("%Y-%m-%d %H:%M")
        
        arrival_times_hours = ensemble[ensemble['target'] == target_names[i]]['arrival time [UT]'].dt.hour
        std_dev_hours = round(np.std(arrival_times_hours), 2)
        
        mean_arrival_speed = ensemble[ensemble['target'] == target_names[i]]['arrival speed [km/s]'].mean()
        median_arrival_speed = ensemble[ensemble['target'] == target_names[i]]['arrival speed [km/s]'].median()
        std_dev_speed = round(np.std(ensemble[ensemble['target'] == target_names[i]]['arrival speed [km/s]']))
        
        #Likelihood of hit at a certain target
        count = (ensemble['target'] == target_names[i]).sum()
        
        tmp_ensemble_results = pd.DataFrame(columns=['target'])
        tmp_ensemble_results.loc[j, 'target'] = target_names[i]
        tmp_ensemble_results['likelihood [%]'] = round(count/(number_of_runs * 0.01))
        
        #pdb.set_trace()
        if no_det_run:
            tmp_ensemble_results['arrival time (det) [UT]'] = 'No fit possible!'
            tmp_ensemble_results['arrival speed (det) [km/s]'] = np.nan
        else:
            if (det_results['target'] == target_names[i]).any():
                #pdb.set_trace()
                tmp_ensemble_results['arrival time (det) [UT]'] = det_results['arrival time [UT]'][j]
                tmp_ensemble_results['arrival speed (det) [km/s]'] = det_results['arrival speed [km/s]'][j]
                #pdb.set_trace()
            else:
                tmp_ensemble_results['arrival time (det) [UT]'] = 'No hit!'
                tmp_ensemble_results['arrival speed (det) [km/s]'] = np.nan
                        
        tmp_ensemble_results['arrival time (mean) [UT]'] = mean_arrival_time
        tmp_ensemble_results['arrival time (median) [UT]'] = median_arrival_time
        tmp_ensemble_results['arrival time (std dev) [h]'] = std_dev_hours
        tmp_ensemble_results['arrival speed (mean) [km/s]'] = round(mean_arrival_speed)
        tmp_ensemble_results['arrival speed (median) [km/s]'] = round(median_arrival_speed)
        tmp_ensemble_results['arrival speed (std dev) [km/s]'] = std_dev_speed
        
        tmp_ensemble_results['dt (mean)'] = round(ensemble[ensemble['target'] == target_names[i]]['dt [h]'].mean(), 2)               
        tmp_ensemble_results['dv (mean)'] = round(ensemble[ensemble['target'] == target_names[i]]['dv [km/s]'].mean(), 2)       
                       
        ensemble_results = pd.concat([ensemble_results, tmp_ensemble_results])
        
        j = j + 1
    
    #pdb.set_trace()
    
    return ensemble_results

def does_cme_hit_elevohi(delta, halfwidth):

    if round(np.rad2deg(np.abs(delta)), 2) < round(np.rad2deg(halfwidth), 2):
        hit = 1
    else:
        hit = 0

    return hit

def does_cme_hit(delta, halfwidth, tol=0.0):
    tol_rad = np.deg2rad(tol)

    return np.abs(delta) < (halfwidth + tol_rad)

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

    n_steps = timesteps.size
    n_ens = gamma.size
    cme_r_ensemble = np.empty((n_steps, n_ens))
    cme_v_ensemble = np.empty((n_steps, n_ens))

    gfac = gamma * 1e-7
    for j in prange(n_ens):
        v_amb = ambient_wind[j]
        v0 = speed_ensemble[j]
        accsign = 1.
        if v0 < v_amb:
            accsign = -1.

        for i in range(n_steps):
            t = timesteps[i]
            term = accsign * gfac[j] * (v0 - v_amb) * t
            cme_r_ensemble[i, j] = (
                (accsign / gfac[j]) * np.log(1 + term)
                + v_amb * t
                + distance0
            )

            # term2 = accsign * gfac[j] * (v0 - v_amb) * t
            cme_v_ensemble[i, j] = (
                (v0 - v_amb) / (1 + term) + v_amb
            )

    return cme_r_ensemble, cme_v_ensemble

def compute_dbm_kinematics_single(tinit, rinit, vinit, gamma, winds):

    AU_in_km = const.au.to_value('km')
    time_step = timedelta(minutes=1)
    timegrid = 1440 * 10  # 10 days with 1 min time step

    time_array = [tinit + i * time_step for i in range(timegrid)]
    tnum = [(t - tinit).total_seconds() for t in time_array]

    vdrag = np.zeros(timegrid)
    rdrag = np.zeros(timegrid)

    accsign = -1 if vinit < winds else 1

    for i in range(timegrid):
        rdrag[i] = (accsign / gamma) * np.log(1 + accsign * gamma * ((vinit - winds) * tnum[i])) + winds * tnum[i] + rinit
        vdrag[i] = (vinit - winds)/ (1 + (accsign * gamma * (vinit - winds) * tnum[i])) + winds

        if not np.isfinite(rdrag[i]):
            raise ValueError("Invalid DBM kinematics")

    R = rdrag / AU_in_km

    return R, vdrag, time_array, tnum

