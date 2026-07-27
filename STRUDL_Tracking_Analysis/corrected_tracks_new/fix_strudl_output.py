import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import glob
import pickle
import os
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
from matplotlib.widgets import RangeSlider
import matplotlib
from utils.helper_functions import get_stereo_heeq_lon
import json

matplotlib.use('Qt5Agg')

def load_jplot(path, instrument, ftpsc, bflag, start, end):

    savepath_jplot = os.path.join(path, 'jplot', ftpsc, bflag, instrument, str(start[:4]), 'plots')
    savepath_param = os.path.join(path, 'jplot', ftpsc, bflag, instrument, str(start[:4]), 'params')

    instrument_name = 'hi1' if instrument == 'hi_1' else 'hi2'

    jplot_name = f"jplot_{instrument_name}_{start}_{end}_{ftpsc}_{bflag[0]}"
    param_fil = glob.glob(f"{savepath_param}/{jplot_name}_params.pkl")[0]

    plot_fil = glob.glob(f"{savepath_jplot}/{jplot_name}.pkl")[0]
    
    
    with open(param_fil, 'rb') as f:
        t_beg, t_end, e_beg, e_end = pickle.load(f)

    with open(plot_fil, 'rb') as f:
        img_rescale, orig = pickle.load(f)

    return img_rescale, orig, t_beg, t_end, e_beg, e_end

class TrackEditor:
    def __init__(self, ax, track_times, track_elongs, datetime_series, strudl_id, donki_feature):
        self.ax = ax
        self.track_times = np.array(track_times)
        self.track_elongs = np.array(track_elongs)
        self.scatter = ax.scatter(track_times, track_elongs, color='deeppink', picker=True, marker='X', alpha=0.7)
        self.datetime_series = datetime_series
        self.strudl_id = strudl_id
        self.donki_feature = donki_feature

        self.selected_idx = None
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_key = ax.figure.canvas.mpl_connect('key_press_event', self.on_key)
        print("Instructions:\n"
              " - Left-click near a point to move it\n"
              " - Right-click to add a new point\n"
              " - Press 'd' to delete selected point\n"
              " - Press 'c' to save track\n")

    def on_click(self, event):

        if not event.inaxes:
            return
        if event.button == 1:  # Left-click: select or move point
            xdata = mdates.date2num(self.track_times)
            ydata = self.track_elongs
            dists = np.hypot(xdata - event.xdata, ydata - event.ydata)
            if len(dists) == 0:
                return
            idx = np.argmin(dists)
            if dists[idx] < 0.3:  # pick tolerance
                self.selected_idx = idx
            else:
                self.selected_idx = None
        elif event.button == 3:  # Right-click: add point
            new_time = mdates.num2date(event.xdata)
            new_elong = event.ydata
            self.track_times = np.append(self.track_times, new_time)
            self.track_elongs = np.append(self.track_elongs, new_elong)
            self._redraw()

    def on_motion(self, event):
        if self.selected_idx is None or not event.inaxes:
            return
        
        self.track_times[self.selected_idx] = mdates.num2date(event.xdata)
        self.track_elongs[self.selected_idx] = event.ydata
        self._redraw()

    def on_release(self, event):
        if self.selected_idx is not None:
            # Sort after releasing the dragged point
            order = np.argsort(self.track_times)
            self.track_times = self.track_times[order]
            self.track_elongs = self.track_elongs[order]
            self.scatter.set_offsets(np.c_[mdates.date2num(self.track_times), self.track_elongs])
            self.ax.figure.canvas.draw_idle()

        self.selected_idx = None

    def on_key(self, event):
        if event.key == 'd' and self.selected_idx is not None:
            self.track_times = np.delete(self.track_times, self.selected_idx)
            self.track_elongs = np.delete(self.track_elongs, self.selected_idx)
            self.selected_idx = None
            self._redraw()

        elif event.key == 'c':
            print("Saving track...")

            # --- Map to nearest discrete times ---
            date_obj_corrected = []
            for p, dat_p in enumerate(self.track_times):
                # elon = self.track_elongs[p]

                arr_ind = (np.abs(self.datetime_series - dat_p.replace(tzinfo=None))).argmin()
                dat_new = self.datetime_series[arr_ind]

                date_obj_corrected.append(dat_new)

            # Convert to DataFrame for saving
            
            df = pd.DataFrame({
                'cmeID_strudl': self.strudl_id,
                'time': date_obj_corrected,
                'elongation': self.track_elongs,
                'featureCode': self.donki_feature
            })

            if not os.path.exists('misc/corrected_tracks'):
                os.makedirs('misc/corrected_tracks')
                
            df.to_csv('misc/corrected_tracks/edited_track_corrected_'+self.strudl_id+'_'+self.donki_feature+'.csv', index=False)
            print(f"Track saved as 'edited_track_corrected_{self.strudl_id}_{self.donki_feature}.csv'")

    def _redraw(self):
        # Make sure all times are naive (no timezone info)
        self.track_times = np.array([t.replace(tzinfo=None) for t in self.track_times])

        self.scatter.set_offsets(np.c_[mdates.date2num(self.track_times), self.track_elongs])
        self.ax.figure.canvas.draw_idle()

def retrack(path_retrack_strudl, path_tracks_elevo):

    ins_list = ['hi_1']
    sc_list = ['A']
    data_list = ['science']
    output_dir = '/Users/maikebauer/Code/ELEvoHI_Python/data/'
    #donki_feature = 'test'

    strudl_redo = json.load(open(path_retrack_strudl, 'r'))

    elevo_redo = json.load(open(path_tracks_elevo, 'r'))

    for j in range(len(strudl_redo)):
        matching_elevo = [i for i in range(len(elevo_redo)) if (elevo_redo[i]['cmeID_elevo'] == strudl_redo[j]['cmeID_elevo']) and (elevo_redo[i]['featureCode'] == strudl_redo[j]['featureCode'])]
        time_elon_elevo = np.array(elevo_redo)[matching_elevo] if len(matching_elevo) > 0 else []

        if len(time_elon_elevo) == 0:
            print(f"No ELEvo track found for STRUDL ID {strudl_redo[j]['cmeID_strudl']} with ELEvo key {strudl_redo[j]['cmeID_elevo']}. Skipping.")
            exit()
            continue
        
        else:
            time_elon_elevo = time_elon_elevo[0]

        start = datetime.datetime.strptime(strudl_redo[j]['time'][0], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d')
        savepath_jplot = os.path.join(output_dir, 'jplot', sc_list[0], data_list[0], ins_list[0], str(start[:4]), 'plots')
        plot_files = sorted(glob.glob(f"{savepath_jplot}/*.pkl"))
        found_flg = False

        for plot_file in plot_files:
            plot_start = plot_file.split('/')[-1].split('_')[2]
            plot_end = plot_file.split('/')[-1].split('_')[3]
            plot_start_dt = datetime.datetime.strptime(plot_start, '%Y%m%d')
            plot_end_dt = datetime.datetime.strptime(plot_end, '%Y%m%d')

            if datetime.datetime.strptime(strudl_redo[j]['time'][0], '%Y-%m-%dT%H:%M:%S.%f') >= plot_start_dt and datetime.datetime.strptime(strudl_redo[j]['time'][-1], '%Y-%m-%dT%H:%M:%S.%f') <= plot_end_dt:
                print(f"Track {j} with start {strudl_redo[j]['time'][0]} and end {strudl_redo[j]['time'][-1]} is within plot {plot_start} to {plot_end}")
                found_flg = True
                break

        if not found_flg:
            print(f"No plot found covering track {j} with start {strudl_redo[j]['time'][0]} and end {strudl_redo[j]['time'][-1]}")
            continue
        img_rescale, orig, t_beg, t_end, e_beg, e_end = load_jplot(output_dir, ins_list[0], sc_list[0], data_list[0],plot_start, plot_end)

        img_rescale = [img_rescale]
        orig = [orig]
        t_beg = [t_beg]
        t_end = [t_end]
        e_beg = [e_beg]
        e_end = [e_end]

        vmins = [np.nanmedian(img) - 0.8 * np.nanstd(img) for img in img_rescale]
        vmaxs = [np.nanmedian(img) + 0.8 * np.nanstd(img) for img in img_rescale]

        cadence_h1 = 40.0

        cadence = [cadence_h1]

        datetime_series = [np.arange(t_beg[0], t_end[0] + datetime.timedelta(minutes=cadence[i]), datetime.timedelta(minutes=cadence[i])).astype(datetime.datetime) for i in range(len(cadence))]

        loc_ticks= int(np.ceil(((t_end[0]-t_beg[0]).total_seconds()/(60*60*24)*1/7)))

        fig, ax = plt.subplots(figsize=(20, 8))

        plt.ylim(np.min(e_beg), np.max(e_end))
        plt.xlim(datetime.datetime.strptime(strudl_redo[j]['time'][0], '%Y-%m-%dT%H:%M:%S.%f')-datetime.timedelta(hours=12), datetime.datetime.strptime(strudl_redo[j]['time'][-1], '%Y-%m-%dT%H:%M:%S.%f')+datetime.timedelta(hours=48))

        plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 24), interval=loc_ticks))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
        plt.gca().xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6), interval=loc_ticks))

        plt.gca().yaxis.set_minor_locator(MultipleLocator(2))

        ax.xaxis_date()

        plt.xlabel('Date (d/m/y)')
        plt.ylabel('Elongation (°)')

        imgs = [img.astype(float) for img in img_rescale]

        # time_axis_elevo_converted = [mdates.datestr2num(t) for t in time_elon_elevo['time_axis [UT]']]
        # time_axis_elevohi_converted = [mdates.datestr2num(t) for t in strudl_redo[j]['time']]

        time_axis_elevo_converted = [datetime.datetime.strptime(t, '%Y-%m-%dT%H:%MZ') for t in time_elon_elevo['time_axis [UT]']]
        time_axis_elevohi_converted = [datetime.datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f') for t in strudl_redo[j]['time']]

        for i in range(len(imgs)):
            im = ax.imshow(imgs[i], cmap='gray', aspect='auto', vmin=vmins[i], vmax=vmaxs[i], interpolation='none', origin=orig[0], extent=[mdates.date2num(t_beg[0] - datetime.timedelta(minutes=cadence[i]/2)), mdates.date2num(t_end[0] + datetime.timedelta(minutes=cadence[i]/2)), e_beg[0], e_end[0]])

            ax.plot(time_axis_elevo_converted, np.array(time_elon_elevo['elongation_apex [°]'])[:,0], color='violet')
            ax.plot(time_axis_elevo_converted, np.array(time_elon_elevo['elongation_apex [°]'])[:,1], color='violet')
            ax.plot(time_axis_elevo_converted, np.array(time_elon_elevo['elongation_apex [°]'])[:,2], color='violet')

            slider_ax = fig.add_axes([0.25, 0.03, 0.55, 0.03])  # x, y, width, height
            contrast_slider = RangeSlider(
                slider_ax, "Contrast",
                vmins[i], vmaxs[i],
                valinit=(vmins[i], vmaxs[i]))
            
            def update_contrast(values):
                vmin, vmax = values
                im.set_clim(vmin=vmin, vmax=vmax)
                fig.canvas.draw_idle()

            contrast_slider.on_changed(update_contrast)

            ax.set_title(strudl_redo[j]['cmeID_strudl'])
            ax.vlines(datetime.datetime.strptime(strudl_redo[j]['time_21_5Rs'], '%Y-%m-%dT%H:%MZ'), np.min(e_beg), np.max(e_end), colors='red', linestyles='dashed')
            _ = TrackEditor(ax, time_axis_elevohi_converted, strudl_redo[j]['elongation'], datetime_series[0], strudl_redo[j]['cmeID_strudl'], strudl_redo[j]['featureCode'])
            plt.show()
    
    plt.close('all')

class TrackCreator:
    def __init__(self, ax, datetime_series, elevo_id, donki_feature):
        self.ax = ax
        self.track_times = np.array([])
        self.track_elongs = np.array([])
        self.datetime_series = datetime_series
        self.elevo_id = elevo_id
        self.donki_feature = donki_feature

        self.scatter = ax.scatter(self.track_times, self.track_elongs, color='deeppink', picker=True, marker='X', alpha=0.7)

        self.selected_idx = None
        self.cid_press = ax.figure.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_release = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_key = ax.figure.canvas.mpl_connect('key_press_event', self.on_key)
        print("Instructions:\n"
              " - Left-click near a point to move it\n"
              " - Right-click to add a new point\n"
              " - Press 'd' to delete selected point\n"
              " - Press 'c' to save track\n")

    def on_click(self, event):

        if not event.inaxes:
            return
        if event.button == 1:  # Left-click: select or move point
            xdata = mdates.date2num(self.track_times)
            ydata = self.track_elongs
            dists = np.hypot(xdata - event.xdata, ydata - event.ydata)
            if len(dists) == 0:
                return
            idx = np.argmin(dists)
            if dists[idx] < 0.3:  # pick tolerance
                self.selected_idx = idx
            else:
                self.selected_idx = None
        elif event.button == 3:  # Right-click: add point
            new_time = mdates.num2date(event.xdata)
            new_elong = event.ydata
            self.track_times = np.append(self.track_times, new_time)
            self.track_elongs = np.append(self.track_elongs, new_elong)
            self._redraw()

    def on_motion(self, event):
        if self.selected_idx is None or not event.inaxes:
            return
        
        self.track_times[self.selected_idx] = mdates.num2date(event.xdata)
        self.track_elongs[self.selected_idx] = event.ydata
        self._redraw()

    def on_release(self, event):
        if self.selected_idx is not None:
            # Sort after releasing the dragged point
            order = np.argsort(self.track_times)
            self.track_times = self.track_times[order]
            self.track_elongs = self.track_elongs[order]
            self.scatter.set_offsets(np.c_[mdates.date2num(self.track_times), self.track_elongs])
            self.ax.figure.canvas.draw_idle()

        self.selected_idx = None

    def on_key(self, event):
        if event.key == 'd' and self.selected_idx is not None:
            self.track_times = np.delete(self.track_times, self.selected_idx)
            self.track_elongs = np.delete(self.track_elongs, self.selected_idx)
            self.selected_idx = None
            self._redraw()

        elif event.key == 'c':
            print("Saving track...")

            # --- Map to nearest discrete times ---
            date_obj_corrected = []
            for p, dat_p in enumerate(self.track_times):
                # elon = self.track_elongs[p]

                arr_ind = (np.abs(self.datetime_series - dat_p.replace(tzinfo=None))).argmin()
                dat_new = self.datetime_series[arr_ind]

                date_obj_corrected.append(dat_new)

            # Convert to DataFrame for saving
            
            df = pd.DataFrame({
                'cmeID_elevo': self.elevo_id,
                'time': date_obj_corrected,
                'elongation': self.track_elongs,
                'featureCode': self.donki_feature
            })

            if not os.path.exists('misc/corrected_tracks'):
                os.makedirs('misc/corrected_tracks')
                
            df.to_csv('misc/corrected_tracks/edited_track_corrected_'+self.elevo_id+'_'+self.donki_feature+'.csv', index=False)
            print(f"Track saved as 'edited_track_corrected_{self.elevo_id}_{self.donki_feature}.csv'")

    def _redraw(self):
        # Make sure all times are naive (no timezone info)
        self.track_times = np.array([t.replace(tzinfo=None) for t in self.track_times])

        self.scatter.set_offsets(np.c_[mdates.date2num(self.track_times), self.track_elongs])
        self.ax.figure.canvas.draw_idle()

def track_new(path_tracks_elevo, path_retrack_strudl=None):

    ins_list = ['hi_1']
    sc_list = ['A']
    data_list = ['science']
    output_dir = '/Users/maikebauer/Code/ELEvoHI_Python/data/'
    #donki_feature = 'test'

    elevo_redo = json.load(open(path_tracks_elevo, 'r'))
    if path_retrack_strudl is not None:
        strudl_redo = json.load(open(path_retrack_strudl, 'r'))
    else:
        strudl_redo = None

    for j in range(len(elevo_redo)):
        matching_strudl = [i for i in range(len(strudl_redo)) if (strudl_redo[i]['cmeID_elevo'] == elevo_redo[j]['cmeID_elevo']) and (strudl_redo[i]['featureCode'] == elevo_redo[j]['featureCode'])]
        
        if len(matching_strudl) > 0:
            print("STRUDL track already exists for ELEvo track with CME ID "+elevo_redo[j]['cmeID_elevo']+" and feature code "+elevo_redo[j]['featureCode']+". Skipping.")
            continue

        start = datetime.datetime.strptime(elevo_redo[j]['tinit [UT]'], '%Y-%m-%dT%H:%MZ').strftime('%Y%m%d')
        savepath_jplot = os.path.join(output_dir, 'jplot', sc_list[0], data_list[0], ins_list[0], str(start[:4]), 'plots')
        plot_files = sorted(glob.glob(f"{savepath_jplot}/*.pkl"))
        found_flg = False

        for plot_file in plot_files:
            plot_start = plot_file.split('/')[-1].split('_')[2]
            plot_end = plot_file.split('/')[-1].split('_')[3]
            plot_start_dt = datetime.datetime.strptime(plot_start, '%Y%m%d')
            plot_end_dt = datetime.datetime.strptime(plot_end, '%Y%m%d')

            if datetime.datetime.strptime(elevo_redo[j]['time_axis [UT]'][0], '%Y-%m-%dT%H:%MZ') >= plot_start_dt and datetime.datetime.strptime(elevo_redo[j]['time_axis [UT]'][0], '%Y-%m-%dT%H:%MZ')+datetime.timedelta(days=3) <= plot_end_dt:
                print(f"Track {j} with start {elevo_redo[j]['time_axis [UT]'][0]} and end {datetime.datetime.strptime(elevo_redo[j]['time_axis [UT]'][0], '%Y-%m-%dT%H:%MZ')+datetime.timedelta(days=3)} is within plot {plot_start} to {plot_end}")
                found_flg = True
                break

        if not found_flg:
            print(f"No plot found covering track {j} with start {elevo_redo[j]['time_axis [UT]'][0]} and end {datetime.datetime.strptime(elevo_redo[j]['time_axis [UT]'][0], '%Y-%m-%dT%H:%MZ')+datetime.timedelta(days=3)}")
            continue
        img_rescale, orig, t_beg, t_end, e_beg, e_end = load_jplot(output_dir, ins_list[0], sc_list[0], data_list[0],plot_start, plot_end)

        img_rescale = [img_rescale]
        orig = [orig]
        t_beg = [t_beg]
        t_end = [t_end]
        e_beg = [e_beg]
        e_end = [e_end]

        vmins = [np.nanmedian(img) - 0.8 * np.nanstd(img) for img in img_rescale]
        vmaxs = [np.nanmedian(img) + 0.8 * np.nanstd(img) for img in img_rescale]

        cadence_h1 = 40.0

        cadence = [cadence_h1]

        datetime_series = [np.arange(t_beg[0], t_end[0] + datetime.timedelta(minutes=cadence[i]), datetime.timedelta(minutes=cadence[i])).astype(datetime.datetime) for i in range(len(cadence))]

        loc_ticks= int(np.ceil(((t_end[0]-t_beg[0]).total_seconds()/(60*60*24)*1/7)))

        fig, ax = plt.subplots(figsize=(20, 8))

        plt.ylim(np.min(e_beg), np.max(e_end))
        plt.xlim(datetime.datetime.strptime(elevo_redo[j]['time_axis [UT]'][0], '%Y-%m-%dT%H:%MZ')-datetime.timedelta(hours=12), datetime.datetime.strptime(elevo_redo[j]['time_axis [UT]'][0], '%Y-%m-%dT%H:%MZ')+datetime.timedelta(hours=48))

        plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 24), interval=loc_ticks))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%y'))
        plt.gca().xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6), interval=loc_ticks))

        plt.gca().yaxis.set_minor_locator(MultipleLocator(2))

        ax.xaxis_date()

        plt.xlabel('Date (d/m/y)')
        plt.ylabel('Elongation (°)')

        imgs = [img.astype(float) for img in img_rescale]

        time_axis_elevo_converted = [datetime.datetime.strptime(t, '%Y-%m-%dT%H:%MZ') for t in elevo_redo[j]['time_axis [UT]']]

        for i in range(len(imgs)):
            im = ax.imshow(imgs[i], cmap='gray', aspect='auto', vmin=vmins[i], vmax=vmaxs[i], interpolation='none', origin=orig[0], extent=[mdates.date2num(t_beg[0] - datetime.timedelta(minutes=cadence[i]/2)), mdates.date2num(t_end[0] + datetime.timedelta(minutes=cadence[i]/2)), e_beg[0], e_end[0]])

            ax.plot(time_axis_elevo_converted, np.array(elevo_redo[j]['elongation_apex [°]'])[:,0], color='violet')
            ax.plot(time_axis_elevo_converted, np.array(elevo_redo[j]['elongation_apex [°]'])[:,1], color='violet')
            ax.plot(time_axis_elevo_converted, np.array(elevo_redo[j]['elongation_apex [°]'])[:,2], color='violet')

            slider_ax = fig.add_axes([0.25, 0.03, 0.55, 0.03])  # x, y, width, height
            contrast_slider = RangeSlider(
                slider_ax, "Contrast",
                vmins[i], vmaxs[i],
                valinit=(vmins[i], vmaxs[i]))
            
            def update_contrast(values):
                vmin, vmax = values
                im.set_clim(vmin=vmin, vmax=vmax)
                fig.canvas.draw_idle()

            contrast_slider.on_changed(update_contrast)

            ax.set_title(elevo_redo[j]['cmeID_elevo'])
            ax.vlines(datetime.datetime.strptime(elevo_redo[j]['tinit [UT]'], '%Y-%m-%dT%H:%MZ'), np.min(e_beg), np.max(e_end), colors='red', linestyles='dashed')
            _ = TrackCreator(ax, datetime_series[0], elevo_redo[j]['cmeID_elevo'], elevo_redo[j]['featureCode'])
            plt.show()
    
    plt.close('all')

def get_corrected_dict(corrected_files, id_key):


    corrected_dict = {}
    for track_file in corrected_files:
        df = pd.read_csv(track_file)
        cme_id = df[id_key].iloc[0]
        times = pd.to_datetime(df['time']).tolist()
        elongs = df['elongation'].tolist()
        fcode = df['featureCode'].iloc[0]
        if cme_id not in corrected_dict:
            corrected_dict[cme_id] = {}

        corrected_dict[cme_id].update({fcode:
            {'time': times,
            'elongation': elongs,
            'featureCode': fcode}})
    
    return corrected_dict

def get_corrected_dict_combined(corrected_files):

    corrected_dict = []
    for track_file in corrected_files:
        df = pd.read_csv(track_file)

        corrected_dict.append({
            'cmeID_strudl': df['cmeID_strudl'].iloc[0],
            'cmeID_elevo': df['cmeID_elevo'].iloc[0],
            'featureCode': df['featureCode'].iloc[0],
            'time':  pd.to_datetime(df['time']).tolist(),
            'elongation': df['elongation'].tolist(),
        })
    
    return corrected_dict

def update_with_corrections(update_dict, corrected_dict, redo_df, id_key):

    if id_key == 'cmeID_strudl':
        time_215_key = 'time_21_5Rs'
        phi_key = 'phi'
        hw_key = 'halfwidth'
        vinit_key = 'vinit'

    elif id_key == 'cmeID_elevo':
        time_215_key = 'tinit [UT]'
        phi_key = 'phi [°]'
        hw_key = 'halfwidth [°]'
        vinit_key = 'vinit [km/s]'

    for j in range(len(redo_df)):
        key = redo_df[j][id_key]
        fcode = redo_df[j]['featureCode']
        if key in corrected_dict:
            if fcode in corrected_dict[key]:
                new_time = np.array([ctim.to_pydatetime() for ctim in corrected_dict[key][fcode]['time']])

                sta_lon_uncorrected = get_stereo_heeq_lon(new_time[0], sc='A')[0]
                sta_lon_corrected = get_stereo_heeq_lon(new_time[0], sc='A')[0]
                corrected_phi = sta_lon_corrected - (sta_lon_uncorrected - redo_df[j][phi_key])

                new_time = np.array([datetime.datetime.strftime(t, '%Y-%m-%dT%H:%M:%S.%f') for t in new_time])

                if id_key != 'cmeID_strudl':
                    strudl_id = None
                    pa = None
                else:
                    strudl_id = redo_df[j]['cmeID_strudl']
                    pa = redo_df[j]['pa']

                update_dict.append({
                    'cmeID_strudl': strudl_id,
                    'cmeID_elevo': redo_df[j]['cmeID_elevo'],
                    'featureCode': redo_df[j]['featureCode'],
                    'L1_ist_obs': redo_df[j]['L1_ist_obs'],
                    'phi': corrected_phi,
                    'vinit': redo_df[j][vinit_key],
                    'halfwidth': redo_df[j][hw_key],
                    'time': new_time,
                    'time_21_5Rs': redo_df[j][time_215_key],
                    'elongation': corrected_dict[key][fcode]['elongation'],
                    'pa': pa
                })
            else:
                if id_key == 'cmeID_strudl':
                    update_dict.append(redo_df[j])
                elif id_key == 'cmeID_elevo':
                    continue

        else:
            if id_key == 'cmeID_strudl':
                update_dict.append(redo_df[j])
            elif id_key == 'cmeID_elevo':
                continue
    
    return update_dict

def update_with_corrections_combined(update_dict, corrected_dict, elevo_df, strudl_df):

    for j in range(len(elevo_df)):
        key = elevo_df[j]['cmeID_elevo']
        fcode = elevo_df[j]['featureCode']

        for entry in corrected_dict:
            if entry['cmeID_elevo'] == key and entry['featureCode'] == fcode:
                new_time = np.array([ctim.to_pydatetime() for ctim in entry['time']])
                sort_order = np.argsort(new_time).astype(int)
                new_time_sorted = np.array(new_time)[sort_order]
                elon_sorted = np.array(entry['elongation'])[sort_order]

                sta_lon_uncorrected = get_stereo_heeq_lon(new_time_sorted[0], sc='A')[0]
                sta_lon_corrected = get_stereo_heeq_lon(new_time_sorted[0], sc='A')[0]
                corrected_phi = sta_lon_corrected - (sta_lon_uncorrected - elevo_df[j]['phi [°]'])


                new_time_sorted = np.array([datetime.datetime.strftime(t, '%Y-%m-%dT%H:%M:%S.%f') for t in new_time_sorted])

                pa = None
                for s in range(len(strudl_df)):
                    if key in strudl_df[s]:
                        pa = strudl_df[s]['pa']

                update_dict.append({
                    'cmeID_strudl': entry['cmeID_strudl'],
                    'cmeID_elevo': entry['cmeID_elevo'],
                    'featureCode': entry['featureCode'],
                    'L1_ist_obs': elevo_df[j]['L1_ist_obs'],
                    'phi': corrected_phi,
                    'vinit': elevo_df[j]['vinit [km/s]'],
                    'halfwidth': elevo_df[j]['halfwidth [°]'],
                    'time': new_time_sorted,
                    'time_21_5Rs': elevo_df[j]['tinit [UT]'],
                    'elongation': elon_sorted,
                    'pa': pa
                })
    
    return update_dict

def create_new_file(path_retrack_strudl, path_tracks_elevo):

    corrected_tracks_path = 'misc/corrected_tracks_new/'
    corrected_tracks_strudl = sorted(glob.glob(f"{corrected_tracks_path}/edited_track_corrected_CME_*.csv"))
    corrected_tracks_elevo = sorted(glob.glob(f"{corrected_tracks_path}/edited_track_corrected_*-CME-*.csv"))
    # read into dictionary with same format as strudl_redo

    corrected_dict_strudl = get_corrected_dict(corrected_tracks_strudl, 'cmeID_strudl')
    corrected_dict_elevo = get_corrected_dict(corrected_tracks_elevo, 'cmeID_elevo')

    strudl_redo = json.load(open(path_retrack_strudl, 'r'))
    elevo_redo = json.load(open(path_tracks_elevo, 'r'))

    updated_strudl_dict = []
    updated_strudl_dict = update_with_corrections(updated_strudl_dict, corrected_dict_strudl, strudl_redo, 'cmeID_strudl')
    updated_strudl_dict = update_with_corrections(updated_strudl_dict, corrected_dict_elevo, elevo_redo, 'cmeID_elevo')

    updated_strudl_dict_df = pd.DataFrame(updated_strudl_dict)
    updated_strudl_dict_df.to_json(path_retrack_strudl.replace('.json', '_corrected.json'), orient='records', date_format='iso')

def create_new_file_combined(path_retrack_strudl, path_tracks_elevo):

    corrected_tracks_path = 'misc/corrected_tracks_new/'
    corrected_tracks = sorted(glob.glob(f"{corrected_tracks_path}/*.csv"))
    # read into dictionary with same format as strudl_redo

    corrected_dict = get_corrected_dict_combined(corrected_tracks)

    strudl_redo = json.load(open(path_retrack_strudl, 'r'))
    elevo_redo = json.load(open(path_tracks_elevo, 'r'))

    updated_dict = []
    update_with_corrections_combined(updated_dict, corrected_dict, elevo_redo, strudl_redo)

    updated_dict_df = pd.DataFrame(updated_dict)
    updated_dict_df.to_json(path_retrack_strudl.replace('.json', '_corrected.json'), orient='records', date_format='iso')

if __name__ == "__main__":

    path_retrack_strudl = '/Users/maikebauer/Code/STRUDL_Tracking_Analysis/data/mean_45_2024_05_01_2025_04_30_strudl_tracks.json'
    path_tracks_elevo = '/Users/maikebauer/Code/ELEvo/data/elevo_results_20240501_20250430.json'

    #track_new(path_tracks_elevo, path_retrack_strudl)
    #retrack(path_retrack_strudl, path_tracks_elevo)
    create_new_file_combined(path_retrack_strudl, path_tracks_elevo)