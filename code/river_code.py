#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 17:19:39 2022

@author: ryaning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib import dates
import numpy as np
import sys

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


#-----------------------figure user options------------------------------------
tick_fontsize = 10
label_fontsize = 12
title_fontsize = 12
mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['axes.linewidth'] = 1.6
plt.rc('xtick', labelsize=tick_fontsize)
plt.rc('ytick', labelsize=tick_fontsize)
plt.rc('axes', labelsize=label_fontsize)
plt.rc('axes', titlesize=title_fontsize)
plt.rc('legend', fontsize=label_fontsize)    # legend fontsize
plt.rc('figure', titlesize=title_fontsize)  # fontsize of the figure title
plt.rcParams['axes.facecolor'] = 'None'

width = 6.88 + 12
height = width/1.618 +1

# =============================================================================
# Import all data and fix timestamps to match each other
# =============================================================================
river_data = pd.read_csv('river_dataframe_localtime.csv')
river_data['date'] = pd.PeriodIndex(river_data['date'], freq='H')
river_data['date'] = river_data['date'].astype(str)
river_data['date'] = pd.to_datetime(river_data['date'].values)
river_data.index = river_data['date'].values

cosipy_data = pd.read_csv('catchment_data_localtime.csv')
cosipy_data['date'] = pd.PeriodIndex(cosipy_data['date'], freq='H')
cosipy_data['date'] = cosipy_data['date'].astype(str)
cosipy_data['date'] = pd.to_datetime(cosipy_data['date'].values, format = '%Y-%m-%d %H:%M')

cosipy_data_mean = pd.read_csv('catchment_data_mean.csv')
cosipy_data_mean['date'] = pd.PeriodIndex(cosipy_data_mean['date'], freq='H')
cosipy_data_mean['date'] = cosipy_data_mean['date'].astype(str)
cosipy_data_mean['date'] = pd.to_datetime(cosipy_data_mean['date'].values, format = '%Y-%m-%d %H:%M')


velocity_data = pd.read_csv('KAN_L_ice_velocity_24h_300722-250822.csv')
velocity_data['date(UTC-2)'] = pd.PeriodIndex(velocity_data['date(UTC-2)'], freq='H')
velocity_data['date(UTC-2)'] = velocity_data['date(UTC-2)'].astype(str)
velocity_data['date(UTC-2)'] = pd.to_datetime(velocity_data['date(UTC-2)'].values)

time_lag = pd.read_csv('time_lag.csv')
time_lag['date'] = pd.to_datetime(time_lag['date'].values, dayfirst=True)
time_lag.index = time_lag['date']
time_lag = time_lag['time_lag'].resample('H').ffill()
print(time_lag)



sdate = pd.to_datetime('2022-08-07 00:00')
edate = pd.to_datetime('2022-08-14 00:00')

water_temp = river_data['water_temp'].rolling(window=20).mean()
river_stage = river_data['river_stage'].rolling(window=20).mean()
river_depth = river_data['river_depth'].rolling(window=20).mean()
turbidity = river_data['SE_volt'].rolling(window=50).mean()

calib_discharge = (river_depth*0.59705) + (10.66038*river_depth**2) + 0.19122

print('average discharge:', np.mean(calib_discharge))

print('max/min discharge:', np.max(calib_discharge), np.min(calib_discharge))

print('stedv', np.std(calib_discharge))


#rain = cosipy_data['RAIN'].values
# =============================================================================
#  Creating big graph figure
# =============================================================================
fig = plt.figure()
spec = gridspec.GridSpec(ncols=1, nrows=5, hspace=0.28)

#----------ax1----temperature and SWin-----------------------------------------
ax1 = fig.add_subplot(spec[0])
lns1 = ax1.plot(cosipy_data['date'], cosipy_data['T2'], label='Air Temperature', color='blue')
ax1b = ax1.twinx()
lns2 = ax1b.plot(cosipy_data['date'], cosipy_data_mean['ME'], label='Available Melt Energy', color='red', linestyle='--')

ax1.set_ylabel('Air Temperature ($^{o}$C)')
ax1b.set_ylabel('Available Melt \nEnergy (W m$^{-2}$)')
ax1.grid(which='major', axis='x', linestyle='--', linewidth=1.0, color='k', zorder=-4)
ax1.set_xlim(sdate, edate)
ax1.set_ylim(1.5,6)
ax1b.set_ylim(0,305)

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, facecolor='white', loc='upper left', bbox_to_anchor=(0.07, 0.999))
     
hours = mdates.HourLocator(interval = 3)
h_fmt = mdates.DateFormatter('%H')

ax1.xaxis.set_minor_locator(hours)
ax1.xaxis.set_minor_formatter(h_fmt)
list_of_axticks = ax1.get_xticks()

hours_2 = mdates.HourLocator(interval = 24)
h_fmt_2 = mdates.DateFormatter('%H')
ax1.xaxis.set_major_locator(hours_2)
ax1.xaxis.set_major_formatter(h_fmt_2)
ax1.tick_params('x', length=1, width=1, which='major')

my_xticks1 = ['7$^{th}$','8$^{th}$','9$^{th}$','10$^{th}$' , '11$^{th}$', '12$^{th}$', '13$^{th}$', '']
num = 19207
n = 0
list_of_axticks = ax1.get_xticks()

for i in range(len(list_of_axticks)):
        print(i)
        ax1.text(list_of_axticks[i] + 0.5, 0.48, my_xticks1[i], 
                size = 10, ha = 'center')
        num = num + 0.5

ax1.text(0.01, 0.83, '(a)', transform=ax1.transAxes, fontsize=20)
ax1.axvspan(xmin=pd.to_datetime('2022-08-03 00:00'), xmax=pd.to_datetime('2022-08-06 00:00'), facecolor='lightblue', alpha=0.3, zorder=-4)
ax1.axvspan(xmin=pd.to_datetime('2022-08-07 00:00'), xmax=pd.to_datetime('2022-08-11 00:00'), facecolor='lightgreen', alpha=0.3, zorder=-4)
ax1.axvspan(xmin=pd.to_datetime('2022-08-11 00:00'), xmax=pd.to_datetime('2022-08-16 00:00'), facecolor='lightgray', alpha=0.4, zorder=-4)


#----------ax2----melt and rain-----------------------------------------
ax2 = fig.add_subplot(spec[1])

lns2 = ax2.bar(cosipy_data['date'].values, cosipy_data_mean['RAIN'], width=1/24, zorder=-4, color='limegreen', edgecolor='k', label='Rainfall')#, label='Rainfall', color='green', linestyle='--')
ax2.set_ylim(0,2)
ax2.set_xlim(sdate, edate)
ax2.set_ylabel('Rainfall (mm)')

ax2b = ax2.twinx()
lns1 = ax2b.plot(cosipy_data['date'].values, cosipy_data['surfM'], label='Surface Melt', color='purple', zorder=5)
ax2b.set_ylabel('Surface Melt (m w.e.)')
ax2.grid(which='major', axis='x', linestyle='--', linewidth=1.0, color='k', zorder=-4)
ax2b.set_ylim(0,4.0)

ax2.legend(loc='upper center', facecolor='white')
lns = lns1
labs = [l.get_label() for l in lns]
ax2b.legend(lns, labs, facecolor='white', loc='upper left', bbox_to_anchor=(0.07, 0.99))
   
hours = mdates.HourLocator(interval = 3)
h_fmt = mdates.DateFormatter('%H')

ax2.xaxis.set_minor_locator(hours)
ax2.xaxis.set_minor_formatter(h_fmt)
list_of_axticks = ax2.get_xticks()

hours_2 = mdates.HourLocator(interval = 24)
h_fmt_2 = mdates.DateFormatter('%H')
ax2.xaxis.set_major_locator(hours_2)
ax2.xaxis.set_major_formatter(h_fmt_2)
ax2.tick_params('x', length=1, width=1, which='major')

my_xticks1 = ['7$^{th}$','8$^{th}$','9$^{th}$','10$^{th}$' , '11$^{th}$', '12$^{th}$', '13$^{th}$', '']
num = 19207
n = 0
list_of_axticks = ax2.get_xticks()

for i in range(len(list_of_axticks)):
        print(i)
        ax2.text(list_of_axticks[i] + 0.5, -0.48, my_xticks1[i], 
                size = 10, ha = 'center')
        num = num + 0.5

ax2.text(0.01, 0.83, '(b)', transform=ax2.transAxes, fontsize=20)
ax2.axvspan(xmin=pd.to_datetime('2022-08-03 00:00'), xmax=pd.to_datetime('2022-08-06 00:00'), facecolor='lightblue', alpha=0.3, zorder=-4)
ax2.axvspan(xmin=pd.to_datetime('2022-08-07 00:00'), xmax=pd.to_datetime('2022-08-11 00:00'), facecolor='lightgreen', alpha=0.3, zorder=-4)
ax2.axvspan(xmin=pd.to_datetime('2022-08-11 00:00'), xmax=pd.to_datetime('2022-08-16 00:00'), facecolor='lightgray', alpha=0.4, zorder=-4)

#----------ax3----discharge and depth-----------------------------------------
ax3 = fig.add_subplot(spec[2])
lns1 = ax3.plot(river_data['date'].values, calib_discharge, label='Discharge', color='red')
ax3b = ax3.twinx()
lns2 = ax3b.plot(time_lag.index, time_lag.values, label='Time Lag', color='k', linestyle='solid')

ax3.set_ylabel('Discharge (m$^{3}$ s$^{-1}$)')
ax3b.set_ylabel('Time Lag (h)')
ax3b.set_ylim(-0.5, 8.5)
ax3.grid(which='major', axis='x', linestyle='--', linewidth=1.0, color='k', zorder=-4)
ax3.set_xlim(sdate, edate)

lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax3.legend(lns, labs, facecolor='white', loc='upper left', bbox_to_anchor=(0.025, 0.95))
   
hours = mdates.HourLocator(interval = 3)
h_fmt = mdates.DateFormatter('%H')

ax3.xaxis.set_minor_locator(hours)
ax3.xaxis.set_minor_formatter(h_fmt)
list_of_axticks = ax3.get_xticks()

hours_2 = mdates.HourLocator(interval = 24)
h_fmt_2 = mdates.DateFormatter('%H')
ax3.xaxis.set_major_locator(hours_2)
ax3.xaxis.set_major_formatter(h_fmt_2)
ax3.tick_params('x', length=1, width=1, which='major')

my_xticks1 = ['7$^{th}$','8$^{th}$','9$^{th}$','10$^{th}$' , '11$^{th}$', '12$^{th}$', '13$^{th}$', '']
num = 19207
n = 0
list_of_axticks = ax3.get_xticks()

for i in range(len(list_of_axticks)):
        print(i)
        ax3.text(list_of_axticks[i] + 0.5, 1.4, my_xticks1[i], 
                size = 10, ha = 'center')
        num = num + 0.5
        
ax3.text(0.01, 0.83, '(c)', transform=ax3.transAxes, fontsize=20)
ax3.axvspan(xmin=pd.to_datetime('2022-08-03 00:00'), xmax=pd.to_datetime('2022-08-06 00:00'), facecolor='lightblue', alpha=0.3, zorder=-4)
ax3.axvspan(xmin=pd.to_datetime('2022-08-07 00:00'), xmax=pd.to_datetime('2022-08-11 00:00'), facecolor='lightgreen', alpha=0.3, zorder=-4)
ax3.axvspan(xmin=pd.to_datetime('2022-08-11 00:00'), xmax=pd.to_datetime('2022-08-16 00:00'), facecolor='lightgray', alpha=0.4, zorder=-4)


#----------ax4----EC and SE_volt-----------------------------------------
ax4 = fig.add_subplot(spec[3])
lns1 = ax4.plot(river_data['date'].values, turbidity, label='Turbidity Proxy', color='darkblue')
ax4.set_ylabel('Voltage (V)')
ax4.grid(which='major', axis='x', linestyle='--', linewidth=1.0, color='k', zorder=-4)
ax4.set_xlim(sdate, edate)

ax4b = ax4.twinx()
lns2 = ax4b.plot(river_data['date'].values, water_temp, label='Water Temperature', color='k', linestyle='--')
ax4b.set_ylabel('Water Temperature ($^{o}$C)')


lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax4.legend(lns, labs, facecolor='white', loc='upper left', bbox_to_anchor=(0.025, 0.95))
   
hours = mdates.HourLocator(interval = 3)
h_fmt = mdates.DateFormatter('%H')

ax4.xaxis.set_minor_locator(hours)
ax4.xaxis.set_minor_formatter(h_fmt)
list_of_axticks = ax4.get_xticks()

hours_2 = mdates.HourLocator(interval = 24)
h_fmt_2 = mdates.DateFormatter('%H')
ax4.xaxis.set_major_locator(hours_2)
ax4.xaxis.set_major_formatter(h_fmt_2)
ax4.tick_params('x', length=1, width=1, which='major')


my_xticks1 = ['7$^{th}$','8$^{th}$','9$^{th}$','10$^{th}$' , '11$^{th}$', '12$^{th}$', '13$^{th}$', '']
num = 19207
n = 0
list_of_axticks = ax4.get_xticks()

for i in range(len(list_of_axticks)):
        print(i)
        ax4.text(list_of_axticks[i] + 0.5, 84.3, my_xticks1[i], 
                size = 10, ha = 'center')
        num = num + 0.5
        n = n + 1

ax4.text(0.01, 0.83, '(d)', transform=ax4.transAxes, fontsize=20)
ax4.axvspan(xmin=pd.to_datetime('2022-08-03 00:00'), xmax=pd.to_datetime('2022-08-06 00:00'), facecolor='lightblue', alpha=0.3, zorder=-4)
ax4.axvspan(xmin=pd.to_datetime('2022-08-07 00:00'), xmax=pd.to_datetime('2022-08-11 00:00'), facecolor='lightgreen', alpha=0.3, zorder=-4)
ax4.axvspan(xmin=pd.to_datetime('2022-08-11 00:00'), xmax=pd.to_datetime('2022-08-16 00:00'), facecolor='lightgray', alpha=0.4, zorder=-4)


#----------ax5----melt and rain-----------------------------------------
# ax5 = fig.add_subplot(spec[4])
# lns1 = ax5.plot(velocity_data['date(UTC-2)'].values, velocity_data['ice_velocity_24h'], label='Discharge', color='red')

# ax5.set_ylabel('Ice Velocity (m day$^{-1}$)')
# ax5.grid(which='major', axis='x', linestyle='--', linewidth=1.0, color='k', zorder=-4)
# ax5.set_xlim(sdate, edate)

# hours = mdates.HourLocator(interval = 3)
# h_fmt = mdates.DateFormatter('%H')

# ax5.xaxis.set_minor_locator(hours)
# ax5.xaxis.set_minor_formatter(h_fmt)
# list_of_axticks = ax5.get_xticks()

# hours_2 = mdates.HourLocator(interval = 24)
# h_fmt_2 = mdates.DateFormatter('%H')
# ax5.xaxis.set_major_locator(hours_2)
# ax5.xaxis.set_major_formatter(h_fmt_2)
# ax5.tick_params('x', length=1, width=1, which='major')

# my_xticks1 = ['7$^{th}$','8$^{th}$','9$^{th}$','10$^{th}$' , '11$^{th}$', '12$^{th}$', '13$^{th}$', '']
# num = 19207
# n = 0
# list_of_axticks = ax5.get_xticks()

# for i in range(len(list_of_axticks)):
#         print(i)
#         ax5.text(list_of_axticks[i] + 0.5, 0.15, my_xticks1[n], 
#                 size = 10, ha = 'center')
#         num = num + 0.5
#         n = n + 1


fig.set_size_inches(width, height)
plt.savefig('big_graph.pdf')
plt.show()
