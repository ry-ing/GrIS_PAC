#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 15:59:35 2022

@author: ryaning
"""

import numpy as np
import xarray as xr
import pypdd
import pandas as pd
import sys
from datetime import timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import dates
import matplotlib as mpl

#MODULES FOR SELECTING INDIVIDUAL GLACIER
import geopandas
import rioxarray
from shapely.geometry import mapping

def glacier_select(DATA, shape_filename):
        print('Clipping DATA to shapefile...')
        glacier_shape = geopandas.read_file(shape_filename)
        DATA.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        DATA.rio.write_crs("epsg:4326", inplace=True)
        DATA = DATA.rio.clip(glacier_shape.geometry.apply(mapping), glacier_shape.crs, drop=False)
        return DATA

def stop():
    print('stopping script...')
    sys.exit()
    
def add_variable_along_latlon(ds, var, name, units, long_name):
    """ This function adds missing variables to the DATA class """
    ds[name] = (('lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

filename = '/Users/ryaning/Documents/PAC/GREENLAND/COSIPY_data/KAN_L_COSIPY_output.nc'
DATA = xr.open_dataset(filename, format='NETCDF4_64BIT')
DATA['time'] = (pd.to_datetime(DATA['time'].values)  - timedelta(hours=2) ).strftime('%Y-%m-%d %H:%M' )   

print(DATA.variables)

# DATA = DATA.resample(time='D').sum()
# DATA['time'] = (pd.to_datetime(DATA['time'].values)).strftime('%Y-%m-%d' ) 
# print(DATA)

#clip to ice only catchment area
shape_filename = '/Users/ryaning/Documents/PAC/GREENLAND/catchment_shapefiles/COSIPY_shapefiles/ice_only/ice_only.shp'
#DATA = glacier_select(DATA, shape_filename)
#DATA.to_netcdf('KAN_L_COSIPY_output_subset.nc')

field_data = {}
def COSIPY_analysis(DATA, sdate, edate, field_list):
    for field in field_list:
        print('Subsetting:', field)
        if field=='RAIN':
            mask = (DATA['time'] > sdate) & (DATA['time'] <= edate)
            masked_field = DATA[field].loc[mask]

            masked_field_sum = masked_field.mean(dim=['lon', 'lat'])
            field_data[field] = masked_field_sum.values 

        elif field=='T2':
            mask = (DATA['time'] > sdate) & (DATA['time'] <= edate)
            masked_field = DATA[field].loc[mask]
            
            masked_field_sum = masked_field.mean(dim=['lon', 'lat'])   
            field_data[field] = masked_field_sum.values - 273.15
            
        elif field=='Q':
            mask = (DATA['time'] > sdate) & (DATA['time'] <= edate)
            masked_field = DATA[field].loc[mask]
            
            masked_field_sum = masked_field.mean(dim=['lon', 'lat'])   
            field_data[field] = masked_field_sum.values 
            
        elif field=='surfM':
            mask = (DATA['time'] > sdate) & (DATA['time'] <= edate)
            masked_field = DATA[field].loc[mask]
            
            masked_field_sum = masked_field.sum(dim=['lon', 'lat'])   
            field_data[field] = masked_field_sum.values 
            
        else:
            mask = (DATA['time'] > sdate) & (DATA['time'] <= edate)
            masked_field = DATA[field].loc[mask]
            
            masked_field_sum = masked_field.mean(dim=['lon', 'lat'])   
            field_data[field] = masked_field_sum.values    
            
            time = DATA['time'].loc[mask].values
    return time, field_data


# def ablation_stake_analysis(DATA, sdate, edate, lat, lon, field, NAME):
#         mask = (DATA['time'] > sdate) & (DATA['time'] <= edate)
#         masked_field = DATA[field].loc[mask]
#         masked_data_sel = masked_field.sel(lon=lon, lat=lat, method='nearest') #selecting model grid point closest to stake
        
#         ice_decrease = (masked_data_sel.sum().values * 100) * 1.1
#         print('%s: %.5f cm' % (NAME, ice_decrease))

ablation_data = {}
def ablation_stake_analysis(DATA, sdate, edate, lats, lons, stakes):
        mask = (DATA['time'] > sdate) & (DATA['time'] <= edate)
        masked_field = DATA['surfM'].loc[mask]
        ablation_data['date'] = DATA['time'].loc[mask]
        
        for stake, x, y in zip(stakes, lons, lats):
            masked_data_sel = masked_field.sel(lon=x, lat=y, method='nearest') #selecting model grid point closest to stake
            ablation_data[stake] = (masked_data_sel.values * 100) * 1.1 #convert to cm of ice
    

sdate = '2022-08-01 00:00'
edate = '2022-08-25 00:00'
field_list = ['Q', 'RAIN', 'T2', 'G', 'surfM', 'N', 'LWin', 'LWout', 'H', 'LE', 'B', 'QRR', 'ME', 'U2', 'EVAPORATION', 'SUBLIMATION', 'CONDENSATION', 'DEPOSITION', 'RH2']
time, field_data = COSIPY_analysis(DATA, sdate, edate, field_list)
time = pd.to_datetime(time)
field_data['date'] = time

catchment_dataframe = pd.DataFrame.from_dict(field_data)
catchment_dataframe.index = (pd.to_datetime(catchment_dataframe['date'].values))#.strftime('%Y-%m-%d' ) 

#catchment_dataframe = catchment_dataframe.resample('D').sum()
#catchment_dataframe['date'] = (pd.to_datetime(catchment_dataframe['date'].values)).strftime('%Y-%m-%d' ) 
print(catchment_dataframe)
print(time)

catchment_dataframe.to_csv('catchment_data_mean.csv')

#------ablation 1--------
# NAME = 'ABL 1'
# sdate = '2022-08-09 13:00'
# edate = '2022-08-13 08:00'

# lat = 67.15390
# lon = -50.03334

#----ablation 2----------
# NAME = 'ABL 2'
# sdate = '2022-08-09 14:00'
# edate = '2022-08-13 08:00'

# lat = 67.15390
# lon = -50.0115

#-----ablation 3---------
# NAME = 'ABL 3'
# sdate = '2022-08-09 14:00'
# edate = '2022-08-13 08:00'

# lat = 67.15694
# lon = -50.00323

#-----ablation 4---------
# NAME = 'ABL 4'
# sdate = '2022-08-09 15:00'
# edate = '2022-08-13 09:00'

# lat = 67.15969
# lon = -49.98653

#-----GPS---------
# NAME = 'GPS'
# sdate = '2022-08-06 18:00'
# edate = '2022-08-13 09:00'

# lat = 67.15339
# lon = -49.97156

# field = 'surfM'
# ablation_stake_analysis(DATA, sdate, edate, lat, lon, field, NAME)





sdate = '2022-08-06 19:00'
edate = '2022-08-14 12:00'

lats = [67.15390, 67.15390,  67.15694, 67.15969,  67.15339]
lons = [-50.03334, -50.0115, -50.00323, -49.98653, -49.97156]
stakes = ['abl_1', 'abl_2', 'abl_3', 'abl_4', 'GPS']


ablation_stake_analysis(DATA, sdate, edate, lats, lons, stakes)
ablation_dataframe = pd.DataFrame.from_dict(ablation_data)

print(ablation_dataframe)

ablation_dataframe.to_csv('cosipy_ablation_data.csv')






# time = pd.to_datetime(time)

# fig, ax = plt.subplots(figsize=(20,10))
# ax.plot(time, field_data['RAIN'], color='b', label='Rainfall')
# ax.plot(time, field_data['Q'], color='r', label='Surface Melt')
# #ax.plot(time, field_data['G'], color='g', label='SWin')

# ax.set_ylabel('(m w.e.)')
# ax.grid()
# ax.set_xlim('2022-08-03 00:00', '2022-08-14 23:00' )
# ax.legend()

# axb = ax.twinx()
# axb.plot(time, field_data['T2'], color='k')
# axb.set_ylabel('Air Temperature ($^{o}$C)', color='k')
# axb.set_ylim(-3, 8)

# hour_locator = mdates.HourLocator(interval=24)
# ax.xaxis.set_major_locator(hour_locator)
# ax.xaxis.set_major_formatter(dates.DateFormatter('%y-%m-%d %H:%M'))

# hour_locator_minor = mdates.HourLocator(interval=3)
# ax.xaxis.set_minor_locator(hour_locator_minor)
# ax.xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))    
# ax.tick_params(axis="x", which="both", rotation=90)

# plt.show()
