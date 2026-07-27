import sys
from PyQt5.QtWidgets import QApplication,QCheckBox, QMainWindow,QSlider, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt, pyqtSignal
import PyQt5
import matplotlib.pyplot as plt
import matplotlib

import pickle
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import glob
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
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['backend'] = 'Qt5Agg' 




def draw_ecliptic(args):
    dtime,diff,header,size,earth,stereo = args
    wcoord = WCS(header)
    # earth = get_body_heliographic_stonyhurst('earth', dtime)
    # stereo = get_horizons_coord('STEREO-A', dtime)
    e_hpc = SkyCoord(earth).transform_to(Helioprojective(observer=stereo))

    e_x = e_hpc.Tx.to(u.deg).value *np.pi/180
    e_y = e_hpc.Ty.to(u.deg).value *np.pi/180


    e_pa = np.arctan2(-np.cos(e_y)*np.sin(e_x), np.sin(e_y))
    delta_pa = e_pa#(-90.0*np.pi/180)#e_pa
    e_val = [(delta_pa)-2*np.pi/180, (delta_pa)+2*np.pi/180]

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
    pa_reg = np.arctan2(-np.cos(ty)*np.sin(tx), np.sin(ty))

    elon_reg = np.arctan2(np.sqrt((np.cos(ty)**2)*(np.sin(tx)**2)+(np.sin(ty)**2)), np.cos(ty)*np.cos(tx))

    elon_reg = resize(elon_reg, (size,size), anti_aliasing=True)           
    pa_reg = resize(pa_reg, (size,size), anti_aliasing=True)

    diff_begin = diff.copy()
    diff_begin = np.repeat(diff_begin[:,:,None],3,2)

    minid = np.abs(pa_reg[:,0]-min(e_val))
    minid = np.argmin(minid)

    maxid = np.abs(pa_reg[:,0]-max(e_val))
    maxid = np.argmin(maxid)
    

    minid2 = np.abs(pa_reg[:,-1]-min(e_val))
    minid2 = np.argmin(minid2)

    maxid2 = np.abs(pa_reg[:,-1]-max(e_val))
    maxid2 = np.argmin(maxid2)

    if header['OBSRVTRY'] == 'STEREO_A':    
        post_conj = int(np.sign(header['crval1']))
    else:
        raise ValueError("Unknown observer in header: {}".format(header['OBSRVTRY']))

    if post_conj:
        tmp = minid
        minid = maxid
        maxid = tmp

        tmp = minid2
        minid2 = maxid2
        maxid2 = tmp
           
    #### displaying lines for the ecliptic in red
    rr, cc = line(maxid,0,maxid2,size-1)
    diff_begin[rr,cc] = [0.0,255.0,0.0]
    diff_begin[rr+1,cc] = [0.0,255.0,0.0]

    rr, cc = line(minid,0,minid2,size-1)
    diff_begin[rr,cc] = [0.0,255.0,0.0]
    diff_begin[rr-1,cc] = [0.0,255.0,0.0]

    return diff_begin, elon_reg

# new_events_list[i][3]
def load_data(event,path_data):
    
    imgs = []
    headers = []
    
    for e in event:
        files_paths = natsorted(glob.glob(path_data+e.strftime("%Y%m%d")+"/science/hi_1/*.fts"))
        for p in files_paths:
            hdul = fits.open(p)
            imgs.append(hdul[0].data)
            headers.append(hdul[0].header)
        
    return imgs,headers



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
    def __init__(self,dates_img,path_rdifs,path_jplot,date_jplot,datetimes_elevo=None,tracks_elevo=None,STRUDL_tracked=None,l1_ist_obs=None, STRUDL_tracked_add=None):
        super().__init__()
        self.setWindowTitle(f"Image and Plot Viewer (ID: {l1_ist_obs})")
        self.size = 600
        self.setGeometry(50, 50, self.size*3, self.size*2)

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
        self.path_jplot = ""
        self.was_terminated=False
        self.images,self.headers = load_data(dates_img,path_rdifs)
        self.header_dates = [datetime.strptime(h["DATE-END"], '%Y-%m-%dT%H:%M:%S.%f') 
                            for h in self.headers]

        
        if self.headers[0]['OBSRVTRY'] == 'STEREO_A':    
            self.post_conj = int(np.sign(self.headers[0]['crval1']))
        else:
            raise ValueError("Unknown observer in header: {}".format(self.headers[0]['OBSRVTRY']))
    
        self.contrast_std2 = 100
        with open(path_jplot+date_jplot.strftime("%Y%m%d")+"_hi_1.p", 'rb') as f:
            jplot_dict = pickle.load(f)

        self.plot_data = jplot_dict["data"]
       
        self.dates = jplot_dict["dates"]
        self.elongations = jplot_dict["elongations"]

        if(self.dates[0].year<2015 or self.dates[0].year>=2025):
            self.origin='upper'
        else:
            self.origin='lower'

        self.medianjplot = np.nanmedian(self.plot_data)
        self.stdjplot = np.nanstd(self.plot_data)


        self.vmin = self.medianjplot - 1.0 * self.stdjplot
        self.vmax = self.medianjplot + 1.0 * self.stdjplot
        
        self.datetimes_elevo = datetimes_elevo
        self.tracks_elevo = tracks_elevo

        beacon_earth = []
        beacon_stereo = []
        for d in range(0,len(self.header_dates),50):
            beacon_earth  += get_body_heliographic_stonyhurst('earth', self.header_dates[d:d+50])
            beacon_stereo += get_horizons_coord('STEREO-A', self.header_dates[d:d+50])
            
        self.elongations_images = []
        self.final_images = []
        # ax.set_xlim(self.event[2],self.event[5])

        args = [(self.header_dates[i],
                self.images[i],
                self.headers[i],
                self.images[0].shape[0],
                beacon_earth[i],
                beacon_stereo[i])
                for i in range(len(self.images))]

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(draw_ecliptic, args))

        for sub_result in results:
            dif, elong1 = sub_result
            self.final_images.append(dif)
            self.elongations_images.append(elong1)
        
        self.pixmap_cache = [None] * len(self.final_images)
        self._build_pixmap_cache()
        self.vmin = np.nanmedian(self.plot_data) - 1 * np.nanstd(self.plot_data)
        self.vmax = np.nanmedian(self.plot_data) + 1 * np.nanstd(self.plot_data)
       
     
        self.tracked_local = []
        self.tracked = []     

        self.tracked_local_add = []
        self.tracked_add = []

        if STRUDL_tracked is not None:
            self.tracked = STRUDL_tracked
            
            for i in range(0,len(STRUDL_tracked)):
                time_track = STRUDL_tracked[i][0]
                elongation = STRUDL_tracked[i][1]*np.pi/180.0
                index = 0
                for i,header_date in enumerate(self.header_dates):
                    if np.abs((header_date.replace(tzinfo=None) - time_track).total_seconds())/60.0 <self.header_match_threshold:
                        index = i
                        break

                elongations = self.elongations_images[index]
                img =   self.final_images[index]
                relative_y = img.shape[0]//2
                relative_x = np.argmin(np.abs(elongations[relative_y,:]-elongation))
                self.tracked_local.append([index, relative_x/img.shape[1],relative_y/img.shape[0]])
            
        if STRUDL_tracked_add is not None:
            self.tracked_add = STRUDL_tracked_add

            for i in range(0,len(STRUDL_tracked_add)):
                time_track = STRUDL_tracked_add[i][0]
                elongation = STRUDL_tracked_add[i][1]*np.pi/180.0
                index = 0

                for i,header_date in enumerate(self.header_dates):
                    if np.abs((header_date.replace(tzinfo=None) - time_track).total_seconds())/60.0 <self.header_match_threshold:
                        index = i
                        break

                elongations = self.elongations_images[index]
                img =   self.final_images[index]
                relative_y = img.shape[0]//2
                relative_x = np.argmin(np.abs(elongations[relative_y,:]-elongation))
                self.tracked_local_add.append([index, relative_x/img.shape[1],relative_y/img.shape[0]])


        self.current_start = len(self.final_images)//2
        self.visible_images = 4

        self.image_labels = []
        for i in range(self.visible_images):
            label = HoverableLabel(self)
            label.setAlignment(Qt.AlignCenter)
            label.hover.connect(self.handle_hover)
            label.click.connect(self.handle_click)
            self.image_labels.append(label)
            self.image_layout.addWidget(label)

        self.figure = plt.figure(figsize=(10, 4))
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
        self.slider_position.setMaximum(len(self.final_images))
        self.slider_position.setValue(len(self.final_images)//2)
        self.slider_position.sliderPosition = len(self.final_images)//2
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
        self.current_image = len(self.final_images)//2
       
        self.update_display()
        self._jplot_background = None

    def _img_to_pixmap(self, img_index):
        """Convert a final_image to a QPixmap, respecting current contrast settings."""
        img = self.final_images[img_index].copy()
        vmin = np.median(img) - 2.5 * np.std(img)
        vmax = np.median(img) + 2.5 * np.std(img)
        img = np.clip(img, vmin, vmax)
        img = (img - vmin) / (vmax - vmin)
        intercept = -(0.5 * self.contrast_std2 / 100) + 0.5
        img = np.clip(self.contrast_std2 / 100 * img + intercept, 0, 1)
        w, h, c = self.final_images[img_index].shape
        qImg = PyQt5.QtGui.QImage(
            (img * 255.0).astype(np.uint8), w, h, 3 * w, QImage.Format_RGB888
        )
        return QPixmap(qImg)

    def _build_pixmap_cache(self):
        for i in range(len(self.final_images)):
            self.pixmap_cache[i] = self._img_to_pixmap(i)

    def _draw_jplot_static(self):
        """Draw the static parts of the jplot and cache the background."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.canvas.draw()
        self._jplot_background = self.canvas.copy_from_bbox(ax.bbox)
        return ax

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
            self.update_display()

        elif(event.button == 1):
            time_event = matplotlib.dates.num2date(event.xdata).replace(tzinfo=None)
            elongation = event.ydata*np.pi/180.0
            index = 0
            for i,header_date in enumerate(self.header_dates):
                if np.abs((header_date.replace(tzinfo=None) - time_event).total_seconds())/60.0 <self.header_match_threshold:
                    index = i
                    break
            elongations = self.elongations_images[index]
           
            img =   self.final_images[index]
            relative_y = img.shape[0]//2
            relative_x = np.argmin(np.abs(elongations[relative_y,:]-elongation))
            self.tracked.append([time_event,event.ydata])
            self.tracked_local.append([index, relative_x/img.shape[1],relative_y/img.shape[0]])
            self.update_display()

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
        self._build_pixmap_cache()
        self.update_display()

    def changeindex(self,value):
        self.current_start = value
        self.update_display()


    def fnc2min(self,params,x,data):
        pw   = params["pw"].value
        adj1 = params['adj1'].value
        adj2 = params['adj2'].value

        model = adj1 * np.power(x+adj2,pw)
        return model- data

    def update_display(self):
        
        self._jplot_background = None
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.vmin = self.medianjplot - self.contrast_std * self.stdjplot
        self.vmax = self.medianjplot + self.contrast_std * self.stdjplot
        
        if(self.medianed):
            disp_Data  = np.median(self.plot_data,1).T
        else:
            a,b,c = self.plot_data.shape
            disp_Data = self.plot_data.reshape((a*b,c)).T

        disp_Data = np.nan_to_num(disp_Data,0.0)

      
        ax.imshow(disp_Data, cmap='gray', aspect='auto',interpolation='none', extent=[self.dates[0], self.dates[-1],self.elongations[0] , self.elongations[1]],vmin=self.vmin,vmax=self.vmax)
        
        if self.datetimes_elevo is not None and self.tracks_elevo is not None:
            ax.plot(self.datetimes_elevo,self.tracks_elevo,color=self.my_palette['my_pink'], alpha=0.7, lw=2, linestyle='--')

        ax.set_ylim(self.elongations[0] , self.elongations[1])


        for i, label in enumerate(self.image_labels):
            image_index = self.current_start + i
            if image_index < len(self.final_images):
                pixmap = self.pixmap_cache[image_index].scaled(
                    self.size, self.size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                
                for t in self.tracked_local:
                    if t[0] == image_index:
                        painter = PyQt5.QtGui.QPainter(pixmap)
                        self._draw_x_marker(
                            painter,
                            int(t[1] * self.size),
                            int(t[2] * self.size),
                            'orange'
                        )
                        painter.end()
                
                for t_add in self.tracked_local_add:
                    if t_add[0] == image_index:
                        painter = PyQt5.QtGui.QPainter(pixmap)
                        pen = PyQt5.QtGui.QPen()
                        pen.setWidth(12)
                        pen.setColor(PyQt5.QtGui.QColor('yellow'))
                        painter.setPen(pen)
                        painter.drawPoint(int(t_add[1] * self.size), int(t_add[2] * self.size))
                        painter.end()

                label.setPixmap(pixmap)
                label.index = image_index
                ax.axvline(x=self.header_dates[image_index], color=[0.0,1.0,0.0], alpha=0.2)
                label.show()
            else:
                label.hide()

        self.prev_button.setEnabled(self.current_start > 0)
        self.next_button.setEnabled(self.current_start + self.visible_images < len(self.final_images))

        
        if(len(self.tracked)>0):
            ax.scatter(np.asarray(self.tracked)[:,0],np.asarray(self.tracked)[:,1], color=self.my_palette['geosphere_orange'], marker='X', alpha=0.9)

        if len(self.tracked_add)>0:
            ax.scatter(np.asarray(self.tracked_add)[:,0],np.asarray(self.tracked_add)[:,1], color=self.my_palette['geosphere_yellow'], marker='s', alpha=0.9)

        loc_ticks= int(np.ceil(((self.dates[-1]-self.dates[0]).total_seconds()/(60*60*24)*1/7)))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 12)))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
        
        #self.canvas.draw()
        self.canvas.draw()
        ax = self.figure.axes[0]
        self._jplot_background = self.canvas.copy_from_bbox(ax.bbox)

    def select_image(self, index):
        self.current_image = index
        self.update_display()

    def handle_click(self, index, relative_x,relative_y,button):
        if(button):
            to_remove = []
            for i in range(0,len(self.tracked_local)):
                distance = np.sqrt((relative_x-self.tracked_local[i][1])**2 + (relative_y-self.tracked_local[i][2])**2)
                if(distance<0.01 and index==self.tracked_local[i][0]):
                    to_remove.append(i)
            self.tracked = [i for j, i in enumerate(self.tracked) if j not in to_remove]
            self.tracked_local = [i for j, i in enumerate(self.tracked_local) if j not in to_remove]
            self.update_display()
        else:
            self.tracked.append([self.header_dates[index]
                                ,self.elongations_images[index][int(relative_y*self.images[0].shape[0]),int(relative_x*self.images[0].shape[0])]*180/np.pi])
            self.tracked_local.append([index, relative_x,relative_y])
            self.update_display()

        
        
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
        ax.draw_artist(dot)
        self.canvas.blit(ax.bbox)
        dot.remove()


    def shift_images_forward(self):
        if self.current_start + self.visible_images < len(self.images):
            self.current_start += 1
            self.slider_position.tracking = True
            self.slider_position.sliderPosition = int(self.current_start)
            self.slider_position.update()
            self.slider_position.repaint()

            self.update_display()

    def shift_images_backward(self):
        if self.current_start > 0:
            self.current_start -= 1
            self.slider_position.tracking = True
            self.slider_position.sliderPosition = int(self.current_start)
            self.slider_position.update()
            self.slider_position.repaint()
            self.update_display()

    def undo_last_click(self):
        self.tracked = self.tracked[:-1]
        self.tracked_local  = self.tracked_local[:-1]
        self.update_display()

    def finish(self):
        print("end")
        self.was_terminated=True
        self.close() 


def tracking(app, dates_img, date_jplot, path_rdifs, path_new_jplot, times, tracks_elevo, times_strudl, tracks_strudl, l1_ist_obs, times_strudl_add=None, tracks_strudl_add=None):

    if times_strudl is None or tracks_strudl is None:
        STRUDL_tracked = None
    else:
        STRUDL_tracked = []

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

    viewer = ImagePlotViewer(dates_img,path_rdifs,path_new_jplot,date_jplot,datetimes_elevo,tracks_elevo,STRUDL_tracked,l1_ist_obs,STRUDL_tracked_add)
    viewer.show()
    app.exec_() 
    # # print(viewer.tracked)
    # sys.exit(app.exec_())
    return viewer.tracked, viewer.was_terminated

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet("QLabel{font-size: 20pt;}\
                      QPushButton{font-size: 20pt;}\
                      QSlider{font-size: 20pt;}\
                      QCheckBox{font-size: 20pt;}")



    path_rdifs = "../HI_Data_Processing/stereo_processed/running_difference/data/A/"
    path_new_jplot = "../HI_Data_Processing/stereo_processed/jplots_pickles/"
    path_elevo = '../ELEvo/data/elevo_results_20240501_20250430.json'
    path_strudl = '../STRUDL_Tracking_Analysis/data/mean_45_2024_05_01_2025_04_30_strudl_tracks.json'
    # path_additional_strudl = None
    
    with open(path_elevo) as json_file:
        data = json.load(json_file)
    
    with open(path_strudl) as json_file_strudl:
        data_strudl = json.load(json_file_strudl)

    # with open(path_additional_strudl) as json_file_strudl_add:
    #     data_strudl_add = json.load(json_file_strudl_add)

    data_strudl_add = []

    for d in data:

        if d['L1_ist_obs'] is None:
            continue

        CME_to_use = d["cmeID_elevo"]

        date_jplot = datetime.strptime(d["tinit [UT]"], "%Y-%m-%dT%H:%MZ")
        date_jplot = datetime(date_jplot.year, date_jplot.month, date_jplot.day)-timedelta(days=1)
        dates_img  = [date_jplot,date_jplot+timedelta(days=1),date_jplot+timedelta(days=2)]

        times = d["time_axis [UT]"]
        tracks_elevo  = d["elongation_apex [\u00b0]"]

        found_flag = 0
        for ds in data_strudl:
            if ds["cmeID_elevo"] == CME_to_use and ds["featureCode"] == d["featureCode"]:
                times_strudl = ds["time"]
                tracks_strudl  = ds["elongation"]
                cme_id_strudl = ds["cmeID_strudl"]
                found_flag = 1

                break 
        
        found_flag_add = 0
        for ds in data_strudl_add:
            if ds["cmeID_elevo"] == CME_to_use and ds["featureCode"] == d["featureCode"]:
                times_strudl_add = ds["time"]
                tracks_strudl_add  = ds["elongation"]
                cme_id_strudl_add = ds["cmeID_strudl"]
                found_flag_add = 1

                break

        if found_flag == 0:
            times_strudl = None
            tracks_strudl = None
            cme_id_strudl = None
        
        if found_flag_add == 0:
            times_strudl_add = None
            tracks_strudl_add = None
            cme_id_strudl_add = None

        # if os.path.isfile(f"../STRUDL_Tracking_Analysis/corrected_tracks_new/{CME_to_use.replace('-', '_').replace('/', '_')}_{cme_id_strudl}_{d['featureCode']}.csv"):
        #     continue

        res, was_terminated = tracking(app, dates_img, date_jplot, path_rdifs, path_new_jplot, times, tracks_elevo, times_strudl, tracks_strudl, CME_to_use+'_'+d["featureCode"], times_strudl_add, tracks_strudl_add)

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