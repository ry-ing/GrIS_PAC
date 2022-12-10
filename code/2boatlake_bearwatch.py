#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 09:07:19 2022

@author: ryaning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.pyplot import cm
import numpy as np
from matplotlib import dates
import matplotlib as mpl
from datetime import timedelta

tick_fontsize = 8
label_fontsize = 10
title_fontsize = 10
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.6
plt.rc('xtick', labelsize=tick_fontsize)
plt.rc('ytick', labelsize=tick_fontsize)
plt.rc('axes', labelsize=label_fontsize)
plt.rc('axes', titlesize=title_fontsize)
plt.rc('legend', fontsize=label_fontsize)    # legend fontsize
plt.rc('figure', titlesize=title_fontsize)  # fontsize of the figure title
plt.rcParams['axes.facecolor'] = 'None'

width = 6.88 + 12
height = width/1.618 - 0.2

def calculate_means(start_array, end_array):
    data = {}
    temp_array = np.zeros(6)
    wind_array = np.zeros(6)
    swin_array = np.zeros(6)
    rain_array = np.zeros(6)

    
    for s, e, i in zip(start_array, end_array, range(6)):
        temp_mask = (MET_DATA['date'] >= s) & (MET_DATA['date'] <= e)
        temp_data = MET_DATA.loc[temp_mask]
        
        temp_mask2 = (RAIN_DATA['date'] >= s) & (RAIN_DATA['date'] <= e)
        temp_data2 = RAIN_DATA.loc[temp_mask2]
        
        temp_array[i] = temp_data['AirTemperature(C)'].mean()
        wind_array[i] = temp_data['WindSpeed(m/s)'].mean()
        swin_array[i] = temp_data['ShortwaveRadiationDown(W/m2)'].mean()
        rain_array[i] = temp_data2['total_precip'].sum()

    
    data['temp'] = temp_array.mean()
    data['wind'] = wind_array.mean()
    data['swin'] = swin_array.mean()
    data['rain'] = rain_array.sum()

    
    return data

BW_times_start = {}
BW_times_end = {}

#----------------------TIME PERIODS OF BEAR WATCH----------------------------------
BW_times_start['JM'] = ['2022-08-03 22:00:00', '2022-08-05 03:00:00', '2022-08-07 01:00:00', '2022-08-08 22:00:00', '2022-08-10 03:00:00', '2022-08-12 01:00:00']
BW_times_end['JM'] = ['2022-08-04 01:00:00', '2022-08-05 06:00:00', '2022-08-07 03:00:00', '2022-08-09 01:00:00', '2022-08-10 06:00:00', '2022-08-12 03:00:00']

BW_times_start['AS'] = ['2022-08-04 01:00:00', '2022-08-05 22:00:00', '2022-08-07 03:00:00', '2022-08-09 01:00:00', '2022-08-10 22:00:00', '2022-08-12 03:00:00']
BW_times_end['AS'] = ['2022-08-04 03:00:00', '2022-08-06 01:00:00', '2022-08-07 06:00:00', '2022-08-09 03:00:00', '2022-08-11 01:00:00', '2022-08-12 06:00:00']

BW_times_start['DL']= ['2022-08-05 01:00:00', '2022-08-06 22:00:00', '2022-08-08 03:00:00', '2022-08-10 01:00:00', '2022-08-11 22:00:00', '2022-08-13 03:00:00']
BW_times_end['DL']= ['2022-08-05 03:00:00', '2022-08-07 01:00:00', '2022-08-08 06:00:00', '2022-08-10 03:00:00', '2022-08-12 01:00:00', '2022-08-13 06:00:00']

BW_times_start['CR'] = ['2022-08-04 22:00:00', '2022-08-06 03:00:00', '2022-08-08 01:00:00', '2022-08-09 22:00:00', '2022-08-11 03:00:00', '2022-08-13 01:00:00']
BW_times_end['CR'] = ['2022-08-05 01:00:00', '2022-08-06 06:00:00', '2022-08-08 03:00:00', '2022-08-10 01:00:00', '2022-08-11 06:00:00', '2022-08-13 03:00:00']

BW_times_start['ES'] = ['2022-08-04 03:00:00', '2022-08-06 01:00:00', '2022-08-07 22:00:00', '2022-08-09 03:00:00', '2022-08-11 01:00:00', '2022-08-12 22:00:00']
BW_times_end['ES'] = ['2022-08-04 06:00:00', '2022-08-06 03:00:00', '2022-08-08 01:00:00', '2022-08-09 06:00:00', '2022-08-11 03:00:00', '2022-08-13 01:00:00']
#------------------------------------------------------------------------------

#-------------------LOAD IN DATA ----------------
Station = 'KAN_B'
met_filename = '/Users/ryaning/Documents/PAC/GREENLAND/PROMICE/%s_august2022.csv' % Station
MET_DATA = pd.read_csv(met_filename, index_col=0)
MET_DATA['date'] = pd.to_datetime(MET_DATA['date'].values) - timedelta(hours=2)

RAIN_DATA = pd.read_csv('/Users/ryaning/Documents/PAC/GREENLAND/ERA_PDD/ERA5_precip_2boatlake.csv', index_col=0)
RAIN_DATA['date'] = pd.to_datetime(RAIN_DATA['date'].values) - timedelta(hours=2)

T2_DATA = pd.read_csv('/Users/ryaning/Documents/PAC/GREENLAND/ERA_PDD/ERA5_t2_2boatlake.csv', index_col=0)
T2_DATA['date'] = pd.to_datetime(T2_DATA['date'].values) - timedelta(hours=2)


#--------MASK TIME TO DURATION OF FIELDWORK-------------------
start_date = '2022-08-03 16:30:00'
end_date = '2022-08-13 12:00:00'
print('\nSelecting data from', start_date, 'to', end_date, '\n')
mask = (MET_DATA['date'] >= start_date) & (MET_DATA['date'] <= end_date)
MET_DATA = MET_DATA.loc[mask]
met_time = MET_DATA['date']




air_temp = MET_DATA['AirTemperature(C)']
cloud = MET_DATA['CloudCover']
wind_speed = MET_DATA['WindSpeed(m/s)']
swin = MET_DATA['ShortwaveRadiationDown(W/m2)']
rain = RAIN_DATA['total_precip']
t2 = T2_DATA['air_temp']

ES_data = calculate_means(BW_times_start['ES'], BW_times_end['ES'])
CR_data = calculate_means(BW_times_start['CR'], BW_times_end['CR'])
DL_data = calculate_means(BW_times_start['DL'], BW_times_end['DL'])
AS_data = calculate_means(BW_times_start['AS'], BW_times_end['AS'])
JM_data = calculate_means(BW_times_start['JM'], BW_times_end['JM'])




#------------------------------------PLOTTING FIGURE-------------------------------------------------
fig = plt.figure()
spec = gridspec.GridSpec(ncols=4, nrows=3, hspace=0.25, wspace=0.38)

ax1 = fig.add_subplot(spec[0,:])
lns1 = ax1.plot(met_time, air_temp, label='Air Temperature', color='blue')
ax1.yaxis.set_ticks(np.arange(0, 12.5, 0.5))
ax1.set_ylim(0,12)
ax1.set_ylabel('Air Temperature ($^{o}$C)')

#ax1.plot(met_time, t2, color='k')

ax1b = ax1.twinx()
lns2 = ax1b.plot(met_time, swin, label='SW$_{IN}$', color='red', linestyle='--')
ax1b.set_ylabel('SW$_{IN}$ (W m$^{-2}$)')
ax1b.set_ylim(-10,750)

color = cm.YlGnBu(np.linspace(0, 1, 5))
name_list = ['ES', 'CR', 'DL', 'AS', 'JM']
full_name_list = ['Lemon Drizzle & Nutty GB', 'Tuna Fish & Digger', 'Pink Wolf & Urban Bear', 'Bald Eagle & Odd Size', 'Mayo Soup & Princess Sparkles']
legend_patch = {}
for c, name, full_name in zip(color, name_list, full_name_list):
    for s, e in zip(BW_times_start[name], BW_times_end[name]):
        plt.axvspan(xmin=s, xmax=e, facecolor=c, alpha=0.5, zorder=-4)
        legend_patch[name] = mpatches.Patch(color=c, label=full_name)
        
plt.legend(handles=[legend_patch['ES'], legend_patch['CR'], legend_patch['DL'], legend_patch['AS'], legend_patch['JM'] ], bbox_to_anchor=(0.026, 1.05))
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0, facecolor='white', bbox_to_anchor=(0.978, 1.25))
      
ax1.set_xlim('2022-08-03 18:00:00', '2022-08-13 12:00:00')
hours = mdates.HourLocator(interval = 3)
h_fmt = mdates.DateFormatter('%H')

ax1.xaxis.set_minor_locator(hours)
ax1.xaxis.set_minor_formatter(h_fmt)
list_of_axticks = ax1.get_xticks()

hours_2 = mdates.HourLocator(interval = 24)
h_fmt_2 = mdates.DateFormatter('%H')
ax1.xaxis.set_major_locator(hours_2)
ax1.xaxis.set_major_formatter(h_fmt_2)
ax1.tick_params('x', length=20, width=1, which='major')

my_xticks1 = ['4$^{th}$','5$^{th}$','6$^{th}$','7$^{th}$' , '8$^{th}$' , '9$^{th}$' , '10$^{th}$', '11$^{th}$', '12$^{th}$', '13$^{th}$', '14$^{th}$']
num = 19207
n = 0
for i in range(len(list_of_axticks)):
        print(i)
        ax1.text(list_of_axticks[i] + 0.5, -1.6, my_xticks1[n], 
                size = 10, ha = 'center')
        num = num + 0.5
        n = n + 1

#--------------------------AX2-------------------------------------------------

ax2 = fig.add_subplot(spec[1,:])
lns1 = ax2.plot(met_time, wind_speed, c='green', label='Wind Speed')
ax2.set_ylabel('Wind Speed (m s$^{-1}$)')

ax2b = ax2.twinx()
lns2 = ax2b.plot(met_time, rain, c='purple', label='Total Precipitation $^{*}$')
ax2b.set_ylabel('Total Precipitation (mm) $^{*}$')

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, facecolor='white', bbox_to_anchor=(0.943, 0.88))
     

for c, name in zip(color, name_list):
    for s, e in zip(BW_times_start[name], BW_times_end[name]):
        plt.axvspan(xmin=s, xmax=e, facecolor=c, alpha=0.5)

ax2.set_xlim('2022-08-03 18:00:00', '2022-08-13 12:00:00')
hours = mdates.HourLocator(interval = 3)
h_fmt = mdates.DateFormatter('%H')

ax2.xaxis.set_minor_locator(hours)
ax2.xaxis.set_minor_formatter(h_fmt)
list_of_axticks = ax2.get_xticks()

hours_2 = mdates.HourLocator(interval = 24)
h_fmt_2 = mdates.DateFormatter('%H')
ax2.xaxis.set_major_locator(hours_2)
ax2.xaxis.set_major_formatter(h_fmt_2)
ax2.tick_params('x', length=20, width=1, which='major')

my_xticks1 = ['4$^{th}$','5$^{th}$','6$^{th}$','7$^{th}$' , '8$^{th}$' , '9$^{th}$' , '10$^{th}$', '11$^{th}$', '12$^{th}$', '13$^{th}$', '14$^{th}$']
num = 19207
n = 0
for i in range(len(list_of_axticks)):
        print(i)
        ax2.text(list_of_axticks[i] + 0.5, -1.6, my_xticks1[n], 
                size = 10, ha = 'center')
        num = num + 0.5
        n = n + 1
    
xposition = pd.date_range(start='2022-08-04 00:00:00', end='2022-08-13 00:00:00', freq='D')
for xc in xposition:
    ax2.axvline(x=xc, color='k', linestyle='--')
    
xposition = pd.date_range(start='2022-08-04 00:00:00', end='2022-08-13 00:00:00', freq='D')
for xc in xposition:
    ax1.axvline(x=xc, color='k', linestyle='--')
    
      
full_name_list = ['Lemon Drizzle & \nNutty GB', 'Tuna Fish & \nDigger', 'Pink Wolf & \nUrban Bear', 'Bald Eagle & \nOdd Size', 'Mayo Soup & \nPrincess Sparkles']
    
ax3 = fig.add_subplot(spec[2,0])
airtemp_arr = [ES_data['temp'], CR_data['temp'], DL_data['temp'], AS_data['temp'], JM_data['temp']]
bar1 = ax3.barh(full_name_list, airtemp_arr, color=color, edgecolor='k')
ax3.bar_label(bar1, label_type='edge', padding=5.5, fmt='%.2f $^{o}C$')
ax3.set_xlabel('Average Air Temperature ($^{o}C$)')
ax3.set_xlim(0,8)

ax4 = fig.add_subplot(spec[2,1])
airwind_arr = [ES_data['wind'], CR_data['wind'], DL_data['wind'], AS_data['wind'], JM_data['wind']]   
bar1 = ax4.barh(full_name_list, airwind_arr, color=color, edgecolor='k')
ax4.bar_label(bar1, label_type='edge', padding=5.5, fmt='%.2f m s$^{-1}$')
ax4.set_xlabel('Average Wind Speed (m s$^{-1}$)')
ax4.set_xlim(0,2.5)

ax5 = fig.add_subplot(spec[2,2])
airswin_arr = [ES_data['swin'], CR_data['swin'], DL_data['swin'], AS_data['swin'], JM_data['swin']]   
bar1 = ax5.barh(full_name_list, airswin_arr, color=color, edgecolor='k')
ax5.bar_label(bar1, label_type='edge', padding=5.5, fmt='%.2f W m$^{-2}$')
ax5.set_xlabel('Average SW$_{IN}$ (W m$^{-2}$)')
ax5.set_xlim(0,14)

ax5 = fig.add_subplot(spec[2,3])
airswin_arr = [ES_data['rain'], CR_data['rain'], DL_data['rain'], AS_data['rain'], JM_data['rain']]   
bar1 = ax5.barh(full_name_list, airswin_arr, color=color, edgecolor='k')
ax5.bar_label(bar1, label_type='edge', padding=5.5, fmt='%.2f mm')
ax5.set_xlabel('Total Bearwatch Rainfall (mm) $^{*}$')
ax5.set_xlim(0,9)
    
fig.suptitle('Two Boat Lake Camp \n3$^{rd}$ to 13$^{th}$ August 2022 \n\nKAN_B AWS data and $\\it{ERA5T~reanalysis^{*}}$', fontsize=16)

fig.set_size_inches(width, height)
#plt.savefig('2boatlake.png')
plt.show()


