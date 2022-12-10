#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:25:18 2022

@author: ryaning
"""

import numpy as np
import xarray as xr
import pandas as pd
from datetime import timedelta

filename = 'ERA5_reanalysis_010822_240822.nc'
DATA = xr.open_dataset(filename, chunks='auto')
print(DATA.variables)
#DATA['time'] = DATA['time'] 
DATA['time'] = (pd.to_datetime(DATA['time'].values)  - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M')  


def ERA5_to_csv(DATA, field, short_name, start_date, end_date, lat, lon, location_name):
    DATA[field][0:7,0,:,:] = DATA[field][0:7,1,:,:] #Sorting out expver levels
    print('\nCreating CSV for:', short_name)
    print('\nSelecting data from', start_date, 'to', end_date, '\n')
    mask = (DATA['time'] >= start_date) & (DATA['time'] <= end_date)
    masked_data = DATA[field].loc[mask]
    time = masked_data['time'].values
    
    masked_data_sel = masked_data.sel(longitude=lon, latitude=lat, method='nearest') #selecting model grid point closest to stake
    
    if field=='tp':
        masked_data_sel = masked_data_sel[:,0].values * 1000 #convert to mm
        
    if field=='sf':
        masked_data_sel = masked_data_sel[:,0].values 
        
    if field=='t2m':
        masked_data_sel = masked_data_sel[:,0].values - 273.15
    # else:
    #     masked_data_sel = masked_data_sel[:,0].values
    
    outfile = 'ERA5_%s_%s.csv' % (short_name, location_name)
    pd.DataFrame({'date': time, short_name : masked_data_sel}).to_csv(outfile)
    print('\nFile saved to:', outfile)


start_date = '2022-08-01 00:00:00'
end_date = '2022-08-25 00:00:00'

#----two boat lake camp----
# lat = 67.124215
# lon= -50.176419

#----KAN_L--------------------
lat = 67.093874
lon= -49.9645452

variable = 'tp'
#tp for total precip
#t2m for airtemp at 2m
#sf for Accumulated snow

variable_long_name = 'PCPT'
file_handle = '010822_250822_KAN_L_localtime'
ERA5_to_csv(DATA, variable, variable_long_name, start_date, end_date, lat, lon, file_handle)












