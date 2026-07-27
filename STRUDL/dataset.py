import torch
from torch.utils.data import Dataset
import numpy as np
from skimage import transform
import datetime
import glob
from astropy.io import fits 
import time 


class RundifSequence_Test(Dataset):
    def __init__(self, data_path,  pair, win_size=16, stride=2, width_par=128):

        start = datetime.datetime.strptime(pair['start'], '%Y_%m_%d')
        end = datetime.datetime.strptime(pair['end'], '%Y_%m_%d')

        self.width_par = width_par
        self.data_path = data_path
        self.win_size = win_size
        self.stride = stride

        # Load filenames whose names are between start and end date
        all_files = np.array(sorted(glob.glob(self.data_path + '*.p')))

        all_files = np.array([file for file in all_files if datetime.datetime.strptime(file.split('/')[-1][:15], '%Y%m%d_%H%M%S') >= start and datetime.datetime.strptime(file.split('/')[-1][:15], '%Y%m%d_%H%M%S') <= end])
        self.files_strided = np.lib.stride_tricks.sliding_window_view(all_files,self.win_size)[::self.stride, :]
        
    def __getitem__(self, index):
       
        # seed = int(index)

        im_all = []

        file_names = list(self.files_strided[index])

        height_par = self.width_par
        im_all = []

        for file in file_names:
            dict_file = np.load(file,allow_pickle=True)
            im = dict_file['data']
            #im = np.load(file,allow_pickle=True)

            if self.width_par != 1024:
                im = transform.resize(im, (self.width_par , height_par), anti_aliasing=True, preserve_range=True)

            im_all.append(im)

        im_all = np.array(im_all)
        return {'image':torch.tensor(im_all).unsqueeze(1), 'names':file_names}
    
    def __len__(self):
        return len(self.files_strided)
    