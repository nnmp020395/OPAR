import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

class load_data: 
    def __init__(self, filepath, canal_name, limites, bck_correction = False):
        file = xr.open_dataset(filepath)
        alt_obs = np.where((file['range'] > limites['alt_obs'][0]) & (file['range'] <= limites['alt_obs'][1]))[0]
        alt_background = np.where((file['range'] > limites['alt_background'][0]) & (file['range'] <= limites['alt_background'][1]))[0]
        background = file.sel(channel = canal_name).isel(range = alt_obs).mean(dim = 'range')['signal']
        self.data = file.sel(channel = canal_name).isel(range = alt_obs)['signal']
#         if len(background.shape) == len(self.data.shape):
#             background = background.reshape(-1,1)
        self.background = background#.values.reshape(-1,1)
        self.time = file.time#.values
        self.range = file.range.isel(range = alt_obs) * 1e3 # metres
        self.option = bck_correction

    def range_corrected_signal(self):
        '''
        Fontionc permet de retrouver un signal corrigé de la distance à l'instrument et du fond ciel 
        Input: 
            signal_raw: raw signal (MHz) without any correction
            opar_range: range in meters 
            opar_alt: altitude in meters
            bck_correction: False = non corriger, True = corriger
        Output:
            Signal corrigé 
        '''
        if self.option == False:
            rcs = self.data * np.square(self.range) #MHz.m^2
        else:
            rcs = (self.data - self.background)*np.square(self.range)
        return rcs

def get_step(altitude_array, new_resolution):
    '''
    resolution and altitude should be the same unit
    '''
    resolution_init = np.abs(altitude_array[3] - altitude_array[2])
    step = np.int(new_resolution / resolution_init)
    return step

class ReshapedMatrice:
    def __init__(self, data, step, axis=0):
        self.axis = axis
        if (axis == 0):
            if len(data.shape) == 1:
                sub_data = [data[n:n+step] for n in range(0, data.shape[axis], step)]
            else:
                sub_data = [data[n:n+step, :] for n in range(0, data.shape[axis], step)]
        else:
            sub_data = [data[:, n:n+step] for n in range(0, data.shape[axis], step)]
        self.data = sub_data
        self.step = step
    
    def count(self):
        counted_data = [((self.data[i] > 0) | ~np.isnan(self.data[i])).sum(axis=self.axis) for i in range(len(self.data))] 
        final_data = np.vstack(counted_data).T
        return final_data
    
    def density(self):
        density_data = [((self.data[i] > 0) | ~np.isnan(self.data[i])).sum(axis=self.axis)/self.step for i in range(len(self.data))] 
        final_data = np.vstack(density_data).T
#         final_data = np.where(density_data < 1, np.nan, 1)
        return final_data

    def close_top(self):
        close_data = [np.nanmax(self.data[i], axis=self.axis) for i in range(len(self.data))] 
        final_data = np.vstack(close_data).T
        return final_data  
    
    def mean(self):
        mean_data = [np.nanmean(self.data[i], axis=self.axis) for i in range(len(self.data))] 
        final_data = np.vstack(mean_data).T
        return final_data 