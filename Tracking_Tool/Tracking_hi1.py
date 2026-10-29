import sys
from PyQt5.QtWidgets import QApplication,QCheckBox, QMainWindow,QSlider, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt, pyqtSignal
import PyQt5
import matplotlib.pyplot as plt
import matplotlib

import pickle
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import glob
import os
from natsort import natsorted
from datetime import datetime, timedelta
from skimage.draw import line
from skimage.transform import  resize
import matplotlib.dates as mdates

from astropy.wcs import WCS
from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst
from sunpy.coordinates import Helioprojective
from sunpy.coordinates.ephemeris import get_horizons_coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
import json 
import pandas as pd
from collections import OrderedDict

matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['backend'] = 'Qt5Agg' 




# The elongation map follows the WCS, which moves by far less than a pixel over a day, so it is
# computed once per day and shared by that day's frames instead of once per frame: the elongation
# a click reads off then differs by 0.003 deg, a seventh of a pixel at the 0.0198 deg/px of HI-1.
# The ecliptic line is still placed from each frame's own WCS, see ecliptic_columns().
_GEOM_CACHE = OrderedDict()
_GEOM_CACHE_MAX_ENTRIES = 6

# Earth/STEREO-A positions at day boundaries, and the last jplot read, kept across CMEs: candidates
# are tracked in time order, so consecutive ones usually share days.
_EPHEM_CACHE = {}
_JPLOT_CACHE = {}


def compute_geometry(header, size):
    """Elongation map (radians) of a HI-1 frame, cached per day and image size."""

    cache_key = (header['DATE-END'][:10], size)

    cached = _GEOM_CACHE.get(cache_key)

    if cached is not None:
        _GEOM_CACHE.move_to_end(cache_key)
        return cached

    wcoord = WCS(header)

    if(size==512):
        newsize = 256
    else:
        newsize = size

    x = np.linspace(0, newsize-1, newsize)
    y = np.linspace(newsize-1, 0, newsize)
    xv, yv = np.meshgrid(x, y)

    thetax, thetay = wcoord.all_pix2world(xv, yv, 0)

    tx = thetax*np.pi/180
    ty = thetay*np.pi/180

    elon_reg = np.arctan2(np.sqrt((np.cos(ty)**2)*(np.sin(tx)**2)+(np.sin(ty)**2)), np.cos(ty)*np.cos(tx))

    # resizing is only needed for the half-resolution grid the 512 pixel images are computed on
    if newsize != size:
        elon_reg = resize(elon_reg, (size,size), anti_aliasing=True)

    _GEOM_CACHE[cache_key] = elon_reg
    _GEOM_CACHE.move_to_end(cache_key)

    while len(_GEOM_CACHE) > _GEOM_CACHE_MAX_ENTRIES:
        _GEOM_CACHE.popitem(last=False)

    return elon_reg


def ecliptic_pa(dtimes):
    """
    Position angle of the ecliptic (Earth) seen from STEREO-A at each of `dtimes`.

    Earth and STEREO-A are only asked for at the day boundaries the frames fall between and the
    result is interpolated, the way ecliptic_cut() in HI_Data_Processing does it: the angle moves
    by 0.0044 deg (0.1 px) over three days, and get_horizons_coord() is a network call.
    """

    days = sorted({datetime(d.year, d.month, d.day) for d in dtimes})
    bounds = days + [days[-1]+timedelta(days=1)]

    missing = [b for b in bounds if b not in _EPHEM_CACHE]

    if missing:
        earth = get_body_heliographic_stonyhurst('earth', missing)
        stereo = get_horizons_coord('STEREO-A', missing)

        for i, b in enumerate(missing):
            e_hpc = SkyCoord(earth.reshape(-1)[i]).transform_to(Helioprojective(observer=stereo.reshape(-1)[i]))
            _EPHEM_CACHE[b] = (e_hpc.Tx.to(u.deg).value*np.pi/180, e_hpc.Ty.to(u.deg).value*np.pi/180)

    knots = [b.timestamp() for b in bounds]
    frames = [d.timestamp() for d in dtimes]

    e_x = np.interp(frames, knots, [_EPHEM_CACHE[b][0] for b in bounds])
    e_y = np.interp(frames, knots, [_EPHEM_CACHE[b][1] for b in bounds])

    return np.arctan2(-np.cos(e_y)*np.sin(e_x), np.sin(e_y))


def ecliptic_columns(header, size):
    """
    Position angle along the first and the last image column. These two columns are all the
    ecliptic line is anchored on, and asking the WCS for 2 of the 1024 columns is cheap enough to
    do per frame, so the line sits exactly where it used to even though the map behind it is
    shared by the whole day.
    """

    wcoord = WCS(header)

    x = np.array([0.0, size-1])
    y = np.linspace(size-1, 0, size)
    xv, yv = np.meshgrid(x, y)

    thetax, thetay = wcoord.all_pix2world(xv, yv, 0)

    tx = thetax*np.pi/180
    ty = thetay*np.pi/180

    return np.arctan2(-np.cos(ty)*np.sin(tx), np.sin(ty))


def compute_ecliptic_rows(pa_cols, e_pa, post_conj):
    """Rows at which the +-2 deg band around the ecliptic crosses the first and the last column."""

    e_val = [e_pa-2*np.pi/180, e_pa+2*np.pi/180]

    minid = np.argmin(np.abs(pa_cols[:,0]-min(e_val)))
    maxid = np.argmin(np.abs(pa_cols[:,0]-max(e_val)))

    minid2 = np.argmin(np.abs(pa_cols[:,-1]-min(e_val)))
    maxid2 = np.argmin(np.abs(pa_cols[:,-1]-max(e_val)))

    if post_conj:
        minid, maxid = maxid, minid
        minid2, maxid2 = maxid2, minid2

    return int(minid), int(maxid), int(minid2), int(maxid2)


def draw_ecliptic(diff, rows, size):
    """The frame as RGB with the ecliptic displayed as two lines in green."""

    diff_begin = np.repeat(diff[:,:,None],3,2)

    minid, maxid, minid2, maxid2 = rows

    rr, cc = line(maxid,0,maxid2,size-1)
    diff_begin[rr,cc] = [0.0,255.0,0.0]
    diff_begin[rr+1,cc] = [0.0,255.0,0.0]

    rr, cc = line(minid,0,minid2,size-1)
    diff_begin[rr,cc] = [0.0,255.0,0.0]
    diff_begin[rr-1,cc] = [0.0,255.0,0.0]

    return diff_begin


def load_jplot(path):
    """
    The jplot pickle, its median view and its contrast statistics. The last file read is kept: the
    pickle is 150 MB and the median costs 0.3 s, and CMEs that share a day would redo both.
    """

    if _JPLOT_CACHE.get("path") != path:
        with open(path, 'rb') as f:
            jplot_dict = pickle.load(f)

        _JPLOT_CACHE.clear()
        _JPLOT_CACHE.update({"path": path,
                             "dict": jplot_dict,
                             "median": np.nan_to_num(np.median(jplot_dict["data"],1).T, nan=0.0),
                             "stats": (np.nanmedian(jplot_dict["data"]), np.nanstd(jplot_dict["data"]))})

    return _JPLOT_CACHE["dict"], _JPLOT_CACHE["median"], _JPLOT_CACHE["stats"]


# new_events_list[i][3]
def load_data(event,path_data):
    
    imgs = []
    headers = []
    
    for e in event:
        files_paths = natsorted(glob.glob(path_data+e.strftime("%Y%m%d")+"/science/hi_1/*.fts"))
        for p in files_paths:
            with fits.open(p) as hdul:
                # the stored >f4 makes every later pass over the frame slower than native float32
                imgs.append(hdul[0].data.astype(np.float32))
                headers.append(hdul[0].header)
        
    return imgs,headers


def parse_donki_time(time_str):
    """DONKI time string to datetime, passing on the None of an entry time DONKI leaves undefined."""

    if time_str is None:
        return None

    return datetime.strptime(time_str,'%Y-%m-%dT%H:%MZ')


def find_jplot_day(date_jplot, path_new_jplot, search_days=3):
    """
    The day whose jplot covers the CME: date_jplot when its jplot was made, otherwise the first
    of the following days in the image window that has one. A jplot spans about five days from
    its own date, so a day later still covers the CME. None means no jplot exists anywhere in
    the window and the CME cannot be retracked.
    """

    for k in range(search_days):
        day = date_jplot+timedelta(days=k)
        if os.path.isfile(path_new_jplot+day.strftime("%Y%m%d")+"_hi_1.p"):
            return day

    return None


def has_hi1_images(dates_img, path_rdifs):
    """
    Whether any of the days has HI-1 images. load_data() contributes nothing for a day inside a
    data gap, so a partially covered window is fine and only an empty one has to be skipped.
    """

    return any(glob.glob(path_rdifs+e.strftime("%Y%m%d")+"/science/hi_1/*.fts") for e in dates_img)



class HoverableLabel(QLabel):
    hover = pyqtSignal(int, float,float)
    click = pyqtSignal(int, float,float,bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, event):
        x = event.pos().x()
        relative_x = x / self.width()
        relative_y = event.pos().y() / self.height()
        self.hover.emit(self.index, relative_x,relative_y)

    def mousePressEvent(self, event):
        x = event.pos().x()
        relative_x = x / self.width()
        relative_y = event.pos().y() / self.height()
        if (event.button()== PyQt5.QtCore.Qt.RightButton):
            right = True
        else:
            right = False
        self.click.emit(self.index, relative_x,relative_y,right)

class ImagePlotViewer(QMainWindow):
    def __init__(self,dates_img,path_rdifs,path_jplot,date_jplot,datetimes_elevo=None,tracks_elevo=None,STRUDL_tracked=None,l1_ist_obs=None, STRUDL_tracked_add=None, hi1_entry=None, hi1_entry_early=None, hi1_entry_late=None):
        super().__init__()
        self.setWindowTitle(f"Image and Plot Viewer (ID: {l1_ist_obs})")
        self.size = 600
        self.setGeometry(50, 50, self.size*3, self.size*2)

        # Get the center point of the primary screen available geometry
        cp = QApplication.primaryScreen().availableGeometry().center()

        # Move widget center to screen center
        qr = self.frameGeometry()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

        self.header_match_threshold = 20.0 # minutes
        self.my_palette = {'geosphere_purple':'#701F80',
                           'geosphere_green':'#75CC41',
                           'geosphere_yellow':'#F9F200',
                           'geosphere_orange':'#F29707',
                           'my_pink':'#FF69B4'}
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.main_layout = QVBoxLayout()
        self.image_layout = QHBoxLayout()
        self.main_layout.addLayout(self.image_layout)

        self._contrast_rebuild_timer = PyQt5.QtCore.QTimer()
        self._contrast_rebuild_timer.setSingleShot(True)
        self._contrast_rebuild_timer.timeout.connect(self._rebuild_and_refresh)

        # builds the pixmaps of the frames just outside the view while nothing else is happening,
        # so that stepping to them costs nothing
        self._prefetch_queue = []
        self._prefetch_timer = PyQt5.QtCore.QTimer()
        self._prefetch_timer.setSingleShot(True)
        self._prefetch_timer.timeout.connect(self._prefetch_step)
        self.path_jplot = ""
        self.was_terminated=False
        self.images,self.headers = load_data(dates_img,path_rdifs)
        self.n_images = len(self.images)
        self.img_size = self.images[0].shape[0]
        self.header_dates = [datetime.strptime(h["DATE-END"], '%Y-%m-%dT%H:%M:%S.%f') 
                            for h in self.headers]

        
        if self.headers[0]['OBSRVTRY'] == 'STEREO_A':    
            self.post_conj = int(np.sign(self.headers[0]['crval1']))
        else:
            raise ValueError("Unknown observer in header: {}".format(self.headers[0]['OBSRVTRY']))
    
        self.contrast_std2 = 100
        jplot_dict, self._disp_median, jplot_stats = load_jplot(path_jplot+date_jplot.strftime("%Y%m%d")+"_hi_1.p")

        self.plot_data = jplot_dict["data"]
        self._disp_full = None
       
        self.dates = jplot_dict["dates"]
        self.elongations = jplot_dict["elongations"]

        if(self.dates[0].year<2015 or self.dates[0].year>=2025):
            self.origin='upper'
        else:
            self.origin='lower'

        self.medianjplot, self.stdjplot = jplot_stats


        self.vmin = self.medianjplot - 1.0 * self.stdjplot
        self.vmax = self.medianjplot + 1.0 * self.stdjplot
        
        self.datetimes_elevo = datetimes_elevo
        self.tracks_elevo = tracks_elevo

        # DONKI's predicted HI-1 FOV entry time, and the window the fastest/slowest plausible
        # DONKI speed puts it in
        self.hi1_entry = hi1_entry
        self.hi1_entry_early = hi1_entry_early
        self.hi1_entry_late = hi1_entry_late

        self.e_pa = ecliptic_pa(self.header_dates)

        # the frames of one day share their elongation map, so the list holds references to the
        # three maps of the window rather than a copy per frame
        self.elongations_images = [compute_geometry(self.headers[i], self.img_size)
                                   for i in range(self.n_images)]

        # the ecliptic rows, the pixmaps and their contrast levels are only needed once a frame is
        # actually shown, and reading the WCS of a frame costs 9 ms
        self.ecliptic_rows = [None] * self.n_images
        self.pixmap_cache = [None] * self.n_images
        self.img_levels = [None] * self.n_images

        # the points of this session, the only ones that are editable and that get saved
        self.tracked_local = []
        self.tracked = []     

        # the STRUDL track and the corrected tracks are shown for reference only
        self.tracked_strudl = STRUDL_tracked if STRUDL_tracked is not None else []
        self.tracked_local_strudl = self._to_local(self.tracked_strudl)

        self.tracked_add = STRUDL_tracked_add if STRUDL_tracked_add is not None else []
        self.tracked_local_add = self._to_local(self.tracked_add)

        self.visible_images = 4
        self.current_start = self._initial_start()

        self.image_labels = []
        for i in range(self.visible_images):
            label = HoverableLabel(self)
            label.setAlignment(Qt.AlignCenter)
            label.hover.connect(self.handle_hover)
            label.click.connect(self.handle_click)
            self.image_labels.append(label)
            self.image_layout.addWidget(label)

        # a bare Figure, not plt.figure(): pyplot would keep every CME's figure, and with it the
        # viewer and its frames, alive for the rest of the run
        self.figure = Figure(figsize=(10, 4))
        self.figure.subplots_adjust(
            top=0.93,
            bottom=0.07,
            left=0.04,
            right=0.99,
            hspace=0.2,
            wspace=0.2
        )
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("button_press_event", self._on_left_click)
        self.main_layout.addWidget(self.canvas)
        self.layout.addLayout(self.main_layout)
        

        self.controls_layout = QVBoxLayout()

        self.button_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.undo_button = QPushButton("Undo")
        self.prev_button.clicked.connect(self.shift_images_backward)
        self.next_button.clicked.connect(self.shift_images_forward)
        self.undo_button.clicked.connect(self.undo_last_click)
        self.button_layout.addWidget(self.prev_button)
        self.button_layout.addWidget(self.next_button)
        self.button_layout.addWidget(self.undo_button)

        self.finish_button = QPushButton("End")
        self.finish_button.clicked.connect(self.finish)
        self.button_layout.addWidget(self.finish_button)

        self.sliders_layout = QHBoxLayout()
        self.slider = QSlider()
        self.slider.setMinimum(5)
        self.slider.setMaximum(250)
        self.slider.setValue(100)
        self.slider.sliderPosition = 100
        self.slider.setOrientation(1)  # 1 is horizontal, 0 is vertical
        self.slider.valueChanged.connect(self.show_value)
        self.contrast_std = 1.0
        self.labelj = QLabel("JPLOT Contrast")
        self.sliders_layout.addWidget(self.labelj)
        self.sliders_layout.addWidget(self.slider)



        self.slider_position = QSlider()
        self.slider_position.setMinimum(0)
        self.slider_position.setMaximum(self.n_images)
        self.slider_position.setValue(self.current_start)
        self.slider_position.sliderPosition = self.current_start
        self.slider_position.setOrientation(1)  # 1 is horizontal, 0 is vertical
        self.slider_position.valueChanged.connect(self.changeindex)
        self.button_layout.addWidget(self.slider_position)

        
        
        self.slider2 = QSlider()
        self.slider2.setMinimum(1)
        self.slider2.setMaximum(5000)
        self.slider2.setValue(100)
        self.slider2.sliderPosition = 100
        self.slider2.setOrientation(1)  # 1 is horizontal, 0 is vertical
        self.slider2.valueChanged.connect(self.show_value2)
        self.labeli = QLabel("Images Contrast")
        self.sliders_layout.addWidget(self.labeli)
        self.sliders_layout.addWidget(self.slider2)
        
        self.Hist = True
        self.Histb = QCheckBox("Hist eq.")
        self.Histb.setChecked(True)
        self.Histb.stateChanged.connect(self.hist_changed)
        self.button_layout.addWidget(self.Histb)



        self.medianed = True
        self.checkbox = QCheckBox("Medianed")
        self.checkbox.setChecked(True)
        self.checkbox.stateChanged.connect(self.checkbox_changed)
        self.button_layout.addWidget(self.checkbox)


    


        self.controls_layout.addLayout(self.button_layout)
        self.controls_layout.addLayout(self.sliders_layout)
        self.layout.addLayout(self.controls_layout)
        self.current_image = self.current_start

        self._jplot_background = None
        self._setup_jplot()
        self.update_display()
        
    def _rows(self, img_index):
        """Where the ecliptic crosses the frame, read from that frame's own WCS on first use."""
        if self.ecliptic_rows[img_index] is None:
            self.ecliptic_rows[img_index] = compute_ecliptic_rows(
                ecliptic_columns(self.headers[img_index], self.img_size),
                self.e_pa[img_index], self.post_conj
            )

        return self.ecliptic_rows[img_index]

    def _to_local(self, track):
        """Where each point of a track falls on its frame, as fractions of the image size."""

        local = []

        for time_track, elongation in track:
            elongation = elongation*np.pi/180.0
            index = 0

            for i,header_date in enumerate(self.header_dates):
                if np.abs((header_date.replace(tzinfo=None) - time_track).total_seconds())/60.0 <self.header_match_threshold:
                    index = i
                    break

            elongations = self.elongations_images[index]
            img =   self.images[index]
            relative_y = img.shape[0]//2
            relative_x = np.argmin(np.abs(elongations[relative_y,:]-elongation))
            local.append([index, relative_x/img.shape[1],relative_y/img.shape[0]])

        return local

    def _initial_start(self):
        """
        First of the visible frames: the frame nearest DONKI's predicted HI-1 entry, placed second
        so that the frame before the CME should appear is visible too. Without a predicted time, or
        with one outside the window, the view falls back to the middle or to the nearest end.
        """

        if self.hi1_entry is None:
            return self.n_images//2

        index = int(np.argmin([abs((d - self.hi1_entry).total_seconds()) for d in self.header_dates]))

        return int(np.clip(index-1, 0, max(self.n_images - self.visible_images, 0)))

    def _img_to_pixmap(self, img_index):
        """Convert a frame to a QPixmap, respecting current contrast settings."""
        img = draw_ecliptic(self.images[img_index], self._rows(img_index), self.img_size)

        # the levels follow from the frame alone, so they outlive a change of the contrast slider
        if self.img_levels[img_index] is None:
            self.img_levels[img_index] = (np.median(img) - 2.5 * np.std(img),
                                          np.median(img) + 2.5 * np.std(img))

        vmin, vmax = self.img_levels[img_index]
        img = np.clip(img, vmin, vmax)
        img = (img - vmin) / (vmax - vmin)
        intercept = -(0.5 * self.contrast_std2 / 100) + 0.5
        img = np.clip(self.contrast_std2 / 100 * img + intercept, 0, 1)
        w = h = self.img_size
        qImg = PyQt5.QtGui.QImage(
            (img * 255.0).astype(np.uint8), w, h, 3 * w, QImage.Format_RGB888
        )
        return QPixmap(qImg).scaled(
            self.size, self.size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    def _pixmap(self, img_index):
        """The frame's pixmap at display size, built the first time the frame is shown."""
        if self.pixmap_cache[img_index] is None:
            self.pixmap_cache[img_index] = self._img_to_pixmap(img_index)

        return self.pixmap_cache[img_index]

    def _disp_data(self):
        """
        The jplot as it is displayed. Both views are kept: the median costs 0.3 s and the full one
        a 150 MB copy, and update_display() runs on every click and every step of a slider.
        """

        if(self.medianed):
            return self._disp_median

        if self._disp_full is None:
            a,b,c = self.plot_data.shape
            self._disp_full = np.nan_to_num(self.plot_data.reshape((a*b,c)).T, nan=0.0)

        return self._disp_full

    def _offsets(self, tracked):
        """Tracked points as scatter offsets."""

        if len(tracked) == 0:
            return np.empty((0,2))

        return np.column_stack([mdates.date2num([t[0] for t in tracked]),
                                np.asarray([float(t[1]) for t in tracked])])

    def _overlay_artists(self):
        """
        The artists drawn on top of the jplot, in matplotlib's own order: by zorder, ties kept in
        the order they were added, so that blitting them lands the same pixels as a full redraw.
        """

        return sorted(self._overlay, key=lambda artist: artist.get_zorder())

    def _blit_overlay(self, extra=None):
        """Draw the moving artists, and optionally the hover marker, over the cached background."""

        if self._jplot_background is None:
            self.canvas.draw_idle()
            return

        ax = self.figure.axes[0]
        self.canvas.restore_region(self._jplot_background)

        for artist in self._overlay_artists():
            ax.draw_artist(artist)

        if extra is not None:
            ax.draw_artist(extra)

        self.canvas.blit(ax.bbox)

    def _on_draw(self, event):
        """Keep the background the moving artists are blitted onto in step with the canvas."""

        self._jplot_background = self.canvas.copy_from_bbox(self.figure.axes[0].bbox)
        self._blit_overlay()

    def _prefetch_step(self):
        """Build one pending pixmap and come back for the next once the tool is idle again."""

        while self._prefetch_queue:
            i = self._prefetch_queue.pop(0)

            if self.pixmap_cache[i] is None:
                self._pixmap(i)
                break

        if self._prefetch_queue:
            self._prefetch_timer.start(0)

    def _setup_jplot(self):
        """
        Draw the jplot and create every artist on it once. update_display() then only updates the
        artists, instead of clearing the figure and drawing all of it again on every interaction.
        """

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        self.jplot_im = ax.imshow(self._disp_data(), cmap='gray', aspect='auto',interpolation='none', extent=[self.dates[0], self.dates[-1],self.elongations[0] , self.elongations[1]],vmin=self.vmin,vmax=self.vmax)

        # everything drawn on top of the jplot itself, collected in the order it is added to the
        # axes so that the blitted overlay can reproduce matplotlib's own drawing order
        self._overlay = []

        if self.datetimes_elevo is not None and self.tracks_elevo is not None:
            self._overlay += ax.plot(self.datetimes_elevo,self.tracks_elevo,color=self.my_palette['my_pink'], alpha=0.85, lw=3, linestyle='--')

        if self.hi1_entry_early is not None and self.hi1_entry_late is not None:
            self._overlay.append(ax.axvspan(self.hi1_entry_early, self.hi1_entry_late, color='red', alpha=0.12))

        if self.hi1_entry is not None:
            self._overlay.append(ax.axvline(self.hi1_entry, color='red', linestyle='--', lw=2))

        ax.set_ylim(self.elongations[0] , self.elongations[1])

        self.frame_lines = [ax.axvline(x=self.dates[0], color=[0.0,1.0,0.0], alpha=0.2)
                            for _ in range(self.visible_images)]

        # created in the order they stack, since the scatters share a zorder: the points of this
        # session are drawn over both reference tracks
        self.tracked_strudl_scatter = ax.scatter([],[], color=self.my_palette['geosphere_yellow'], marker='s', alpha=0.9)
        self.tracked_add_scatter = ax.scatter([],[], color=self.my_palette['geosphere_orange'], marker='X', alpha=0.9)
        self.tracked_scatter = ax.scatter([],[], color='red', marker='X', alpha=0.9)

        self._overlay += self.frame_lines + [self.tracked_strudl_scatter, self.tracked_add_scatter,
                                             self.tracked_scatter]

        # the frame of the axes sits above all of them, so it has to be part of the overlay too,
        # otherwise a blitted line would be drawn over it instead of under it
        self._overlay += list(ax.spines.values())

        # everything above the jplot is drawn over the cached background instead of through the
        # canvas, which keeps a step or a click at about a millisecond
        for artist in self._overlay:
            artist.set_animated(True)

        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 12)))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))

        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.draw()

    def _draw_x_marker(self, painter, x, y, color_name, arm=6, width=5):
        """Draw a thick pyplot-like 'X' marker at pixel coordinates."""
        pen = PyQt5.QtGui.QPen()
        pen.setWidth(width)
        pen.setColor(PyQt5.QtGui.QColor(color_name))
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawLine(x - arm, y - arm, x + arm, y + arm)
        painter.drawLine(x - arm, y + arm, x + arm, y - arm)


    def hist_changed(self,state):
        self.Hist = bool(state)
        self.update_display()

    def _on_left_click(self,event):
        if(event.button == 3):
            to_remove=[]
            for i in range(0,len(self.tracked)):
                date = self.tracked[i][0].replace(tzinfo=None)
                if(np.abs((date-matplotlib.dates.num2date(event.xdata).replace(tzinfo=None)).total_seconds())/60.0 <15.0 and np.abs(self.tracked[i][1]- event.ydata)<0.5):
                    to_remove.append(i)
            self.tracked = [i for j, i in enumerate(self.tracked) if j not in to_remove]
            self.tracked_local = [i for j, i in enumerate(self.tracked_local) if j not in to_remove]
            self.update_display(redraw_jplot=False)

        elif(event.button == 1):
            time_event = matplotlib.dates.num2date(event.xdata).replace(tzinfo=None)
            elongation = event.ydata*np.pi/180.0
            index = 0
            for i,header_date in enumerate(self.header_dates):
                if np.abs((header_date.replace(tzinfo=None) - time_event).total_seconds())/60.0 <self.header_match_threshold:
                    index = i
                    break
            elongations = self.elongations_images[index]
           
            img =   self.images[index]
            relative_y = img.shape[0]//2
            relative_x = np.argmin(np.abs(elongations[relative_y,:]-elongation))
            self.tracked.append([time_event,event.ydata])
            self.tracked_local.append([index, relative_x/img.shape[1],relative_y/img.shape[0]])
            self.update_display(redraw_jplot=False)

    def show_value(self,value):
        self.contrast_std = value/100
        self.update_display()

    def checkbox_changed(self,state):
        self.medianed = bool(state)
        self.update_display()

    def show_value2(self, value):
        self.contrast_std2 = value
        self._contrast_rebuild_timer.start(150)  # restart timer on each slider move

    def _rebuild_and_refresh(self):
        # only the frames that are on screen are rebuilt, the rest when they are scrolled to
        self.pixmap_cache = [None] * self.n_images
        self.update_display()

    def changeindex(self,value):
        self.current_start = value
        self.update_display(redraw_jplot=False)


    def fnc2min(self,params,x,data):
        pw   = params["pw"].value
        adj1 = params['adj1'].value
        adj2 = params['adj2'].value

        model = adj1 * np.power(x+adj2,pw)
        return model- data

    def update_display(self, redraw_jplot=True):

        # the jplot and its artists already exist, so only what changed is updated. Moving the view
        # or changing tracked points leaves everything behind the overlay untouched, and is blitted
        # instead of costing a redraw of the whole canvas.
        if redraw_jplot:
            self._jplot_background = None

        self.vmin = self.medianjplot - self.contrast_std * self.stdjplot
        self.vmax = self.medianjplot + self.contrast_std * self.stdjplot

        self.jplot_im.set_data(self._disp_data())
        self.jplot_im.set_clim(self.vmin, self.vmax)

        for i, label in enumerate(self.image_labels):
            image_index = self.current_start + i
            if image_index < self.n_images:
                # a copy, so the markers are not painted into the cached pixmap
                pixmap = QPixmap(self._pixmap(image_index))

                for t_strudl in self.tracked_local_strudl:
                    if t_strudl[0] == image_index:
                        painter = PyQt5.QtGui.QPainter(pixmap)
                        pen = PyQt5.QtGui.QPen()
                        pen.setWidth(12)
                        pen.setColor(PyQt5.QtGui.QColor('yellow'))
                        painter.setPen(pen)
                        painter.drawPoint(int(t_strudl[1] * self.size), int(t_strudl[2] * self.size))
                        painter.end()

                for t_add in self.tracked_local_add:

                    if t_add[0] == image_index:
                        painter = PyQt5.QtGui.QPainter(pixmap)
                        self._draw_x_marker(
                            painter,
                            int(t_add[1] * self.size),
                            int(t_add[2] * self.size),
                            'orange'
                        )
                        painter.end()

                for t in self.tracked_local:

                    if t[0] == image_index:
                        painter = PyQt5.QtGui.QPainter(pixmap)
                        self._draw_x_marker(
                            painter,
                            int(t[1] * self.size),
                            int(t[2] * self.size),
                            'red'
                        )
                        painter.end()

                label.setPixmap(pixmap)
                label.index = image_index
                self.frame_lines[i].set_xdata([mdates.date2num(self.header_dates[image_index])]*2)
                self.frame_lines[i].set_visible(True)
                label.show()
            else:
                self.frame_lines[i].set_visible(False)
                label.hide()

        self.prev_button.setEnabled(self.current_start > 0)
        self.next_button.setEnabled(self.current_start + self.visible_images < self.n_images)

        self.tracked_strudl_scatter.set_offsets(self._offsets(self.tracked_strudl))
        self.tracked_add_scatter.set_offsets(self._offsets(self.tracked_add))
        self.tracked_scatter.set_offsets(self._offsets(self.tracked))

        if redraw_jplot:
            self.canvas.draw_idle()
        else:
            self._blit_overlay()

        # the frames around the view, nearest and ahead of the view first
        self._prefetch_queue = [i for i in [self.current_start+self.visible_images+k for k in range(4)]
                                        + [self.current_start-1-k for k in range(3)]
                                if 0 <= i < self.n_images and self.pixmap_cache[i] is None]

        if self._prefetch_queue:
            self._prefetch_timer.start(0)

    def select_image(self, index):
        self.current_image = index
        self.update_display(redraw_jplot=False)

    def handle_click(self, index, relative_x,relative_y,button):
        if(button):
            to_remove = []
            for i in range(0,len(self.tracked_local)):
                distance = np.sqrt((relative_x-self.tracked_local[i][1])**2 + (relative_y-self.tracked_local[i][2])**2)
                if(distance<0.01 and index==self.tracked_local[i][0]):
                    to_remove.append(i)
            self.tracked = [i for j, i in enumerate(self.tracked) if j not in to_remove]
            self.tracked_local = [i for j, i in enumerate(self.tracked_local) if j not in to_remove]
            self.update_display(redraw_jplot=False)
        else:
            self.tracked.append([self.header_dates[index]
                                ,self.elongations_images[index][int(relative_y*self.images[0].shape[0]),int(relative_x*self.images[0].shape[0])]*180/np.pi])
            self.tracked_local.append([index, relative_x,relative_y])
            self.update_display(redraw_jplot=False)

        
        
    def get_index(self,date):
        for d in self.dates:
            if(np.abs((d-date).total_seconds())<200):
                return d

    def handle_hover(self, index, relative_x, relative_y):
        if self._jplot_background is None:
            return
        ax = self.figure.axes[0]
        self.canvas.restore_region(self._jplot_background)
        
        y = self.elongations_images[index][
            int(relative_y * self.images[0].shape[0]),
            int(relative_x * self.images[0].shape[0])
        ] * 180/np.pi
        
        dot = ax.scatter(
            self.header_dates[index],
            y, color='r', animated=True
        )
        self._blit_overlay(extra=dot)
        dot.remove()


    def shift_images_forward(self):
        if self.current_start + self.visible_images < self.n_images:
            self.current_start += 1
            self.slider_position.tracking = True
            self.slider_position.sliderPosition = int(self.current_start)
            self.slider_position.update()
            self.slider_position.repaint()

            self.update_display(redraw_jplot=False)

    def shift_images_backward(self):
        if self.current_start > 0:
            self.current_start -= 1
            self.slider_position.tracking = True
            self.slider_position.sliderPosition = int(self.current_start)
            self.slider_position.update()
            self.slider_position.repaint()
            self.update_display(redraw_jplot=False)

    def undo_last_click(self):
        self.tracked = self.tracked[:-1]
        self.tracked_local  = self.tracked_local[:-1]
        self.update_display(redraw_jplot=False)

    def finish(self):
        print("end")
        self.was_terminated=True
        self._prefetch_timer.stop()
        self.close() 


def tracking(app, dates_img, date_jplot, path_rdifs, path_new_jplot, times, tracks_elevo, times_strudl, tracks_strudl, l1_ist_obs, times_strudl_add=None, tracks_strudl_add=None, hi1_entry=None, hi1_entry_early=None, hi1_entry_late=None):

    if times_strudl is None or tracks_strudl is None:
        STRUDL_tracked = None
    else:
        STRUDL_tracked = []

    # a CME that is only retracked because STRUDL tracked it has no ELEvo run to draw
    if times is None:
        datetimes_elevo = None
    else:
        datetimes_elevo = []
        for t in times:
            datetimes_elevo.append(datetime.strptime(t,'%Y-%m-%dT%H:%MZ'))

    if STRUDL_tracked is not None:

        datetimes_strudl = []
        for t in times_strudl:
            datetimes_strudl.append(datetime.strptime(datetime.strptime(t,'%Y-%m-%dT%H:%M:%S.%f').strftime('%Y-%m-%dT%H:%MZ'),'%Y-%m-%dT%H:%MZ'))

        STRUDL_tracked = []
        for i in range(len(datetimes_strudl)):
            STRUDL_tracked.append([datetimes_strudl[i], np.array(tracks_strudl[i])])

    if times_strudl_add is not None and tracks_strudl_add is not None:
        
        datetimes_strudl_add = []
        for t in times_strudl_add:
            datetimes_strudl_add.append(datetime.strptime(datetime.strptime(t,'%Y-%m-%dT%H:%M:%S.%f').strftime('%Y-%m-%dT%H:%MZ'),'%Y-%m-%dT%H:%MZ'))

        STRUDL_tracked_add = []
        for i in range(len(datetimes_strudl_add)):
            STRUDL_tracked_add.append([datetimes_strudl_add[i], np.array(tracks_strudl_add[i])])

    else:
        STRUDL_tracked_add = []

    viewer = ImagePlotViewer(dates_img,path_rdifs,path_new_jplot,date_jplot,datetimes_elevo,tracks_elevo,STRUDL_tracked,l1_ist_obs,STRUDL_tracked_add,hi1_entry,hi1_entry_early,hi1_entry_late)
    viewer.show()
    app.exec_() 
    # # print(viewer.tracked)
    # sys.exit(app.exec_())
    return viewer.tracked, viewer.was_terminated

if __name__ == '__main__':

    SHOW_CORRECTED_TRACKS = False

    app = QApplication(sys.argv)

    app.setStyleSheet("QLabel{font-size: 20pt;}\
                      QPushButton{font-size: 20pt;}\
                      QSlider{font-size: 20pt;}\
                      QCheckBox{font-size: 20pt;}")

    path_rdifs = "../HI_Data_Processing/stereo_processed/running_difference/data/A/"
    path_new_jplot = "../HI_Data_Processing/stereo_processed/jplots_pickles/"
    path_candidates = '../STRUDL_Tracking_Analysis/data/DONKI_cme_analyses_matched.json'
    path_elevo = '../ELEvo/data/elevo_results_20240501_20250501.json'
    path_strudl = '../STRUDL_Tracking_Analysis/data/mean_45_2024_05_01_2025_05_01_strudl_tracks.json'
    path_additional_strudl = '../STRUDL_Tracking_Analysis/data/mean_45_2024_05_01_2025_05_01_strudl_tracks_corrected.json'
    
    with open(path_candidates) as json_file_candidates:
        candidates = [cme for cme in json.load(json_file_candidates) if cme.get('isCandidate')]

    with open(path_elevo) as json_file:
        data = json.load(json_file)
    
    with open(path_strudl) as json_file_strudl:
        data_strudl = json.load(json_file_strudl)

    if SHOW_CORRECTED_TRACKS:
        with open(path_additional_strudl) as json_file_strudl_add:
            data_strudl_add = json.load(json_file_strudl_add)
    else:
        data_strudl_add = []

    elevo_by_key      = {(ds["cmeID_elevo"], ds["featureCode"]): ds for ds in data}
    strudl_by_key     = {(ds["cmeID_elevo"], ds["featureCode"]): ds for ds in data_strudl}
    strudl_add_by_key = {(ds["cmeID_elevo"], ds["featureCode"]): ds for ds in data_strudl_add}

    for d in sorted(candidates, key=lambda c: (c["associatedActivityID"], c["featureCode"])):

        CME_to_use = d["associatedActivityID"]
        key = (CME_to_use, d["featureCode"])

        hi1_entry       = parse_donki_time(d["predictedFirstHI1Atime"])
        hi1_entry_early = parse_donki_time(d["predictedFirstHI1AtimeEarly"])
        hi1_entry_late  = parse_donki_time(d["predictedFirstHI1AtimeLate"])

        # without an entry time DONKI never gets the CME into the FOV, so there is nothing to
        # retrack and no time to anchor the images on
        if hi1_entry is None:
            print(f"Skipping {CME_to_use}: no predicted HI-1 FOV entry time")
            continue

        date_jplot = datetime(hi1_entry.year, hi1_entry.month, hi1_entry.day)-timedelta(days=1)

        jplot_day = find_jplot_day(date_jplot, path_new_jplot)

        if jplot_day is None:
            print(f"Skipping {CME_to_use}: no jplot from {date_jplot.strftime('%Y%m%d')} onwards")
            continue

        # a jplot that was not made means that day is inside a HI-1 data gap, so fall through to
        # the first day that has one and show the images from there
        date_jplot = jplot_day
        dates_img  = [date_jplot,date_jplot+timedelta(days=1),date_jplot+timedelta(days=2)]

        if not has_hi1_images(dates_img, path_rdifs):
            print(f"Skipping {CME_to_use}: no HI-1 images from {date_jplot.strftime('%Y%m%d')} onwards")
            continue

        #elevo = elevo_by_key.get(key)
        elevo = None
        
        if elevo is None:
            times = None
            tracks_elevo = None
        else:
            times = elevo["time_axis [UT]"]
            tracks_elevo  = elevo["elongation_apex [\u00b0]"]


        strudl = strudl_by_key.get(key)

        if strudl is None:
            times_strudl = None
            tracks_strudl = None
            # a CME STRUDL matched but with no track at the Earth position angle still has an id
            cme_id_strudl = d["associatedStrudlID"]
        else:
            times_strudl = strudl["time"]
            tracks_strudl  = strudl["elongation"]
            cme_id_strudl = strudl["cmeID_strudl"]

        strudl_add = strudl_add_by_key.get(key)

        if strudl_add is None:
            times_strudl_add = None
            tracks_strudl_add = None
            cme_id_strudl_add = None
        else:
            times_strudl_add = strudl_add["time"]
            tracks_strudl_add  = strudl_add["elongation"]
            cme_id_strudl_add = strudl_add["cmeID_strudl"]

        # if os.path.isfile(f"../STRUDL_Tracking_Analysis/corrected_tracks_new/{CME_to_use.replace('-', '_').replace('/', '_')}_{cme_id_strudl}_{d['featureCode']}.csv"):
        #     continue

        res, was_terminated = tracking(app, dates_img, date_jplot, path_rdifs, path_new_jplot, times, tracks_elevo, times_strudl, tracks_strudl, CME_to_use+'_'+d["featureCode"], times_strudl_add, tracks_strudl_add, hi1_entry, hi1_entry_early, hi1_entry_late)

        tracked_dates = []
        tracked_elongs = []

        for element in res:
            tracked_dates.append(element[0])
            tracked_elongs.append(element[1])
    
        
        sort_order = np.argsort(tracked_dates).astype(int)
        sorted_dates = np.array(tracked_dates)[sort_order]
        sorted_elon = np.array(tracked_elongs)[sort_order]

        if len(np.unique(sorted_dates)) != len(sorted_dates):
            sorted_dates_unique = []
            sorted_elon_unique =  []

            unique_dates, unique_counts = np.unique(sorted_dates, return_counts=True)

            for i in range(len(unique_dates)):
                sorted_dates_unique.append(unique_dates[i])

                if unique_counts[i] > 1:
                    sorted_elon_unique.append(np.mean(sorted_elon[np.where(sorted_dates == unique_dates[i])]))
                else:
                    sorted_elon_unique.append(sorted_elon[i])

            sorted_dates_unique = np.array(sorted_dates_unique)
            sorted_elon_unique = np.array(sorted_elon_unique)
        
        else:
            sorted_dates_unique = np.copy(sorted_dates)
            sorted_elon_unique = np.copy(sorted_elon)

        df = pd.DataFrame({
            "cmeID_elevo": CME_to_use,
            "cmeID_strudl": cme_id_strudl,
            "featureCode": d["featureCode"],
            "time": sorted_dates_unique,
            "elongation": sorted_elon_unique
        })

        if was_terminated:
            df.to_csv(f"../STRUDL_Tracking_Analysis/corrected_tracks_new/{CME_to_use.replace('-', '_').replace('/', '_')}_{cme_id_strudl}_{d['featureCode']}.csv", index=False)