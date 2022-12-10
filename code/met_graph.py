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

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w


#-----------------------figure user options------------------------------------
tick_fontsize = 10
label_fontsize = 10
title_fontsize = 12
mpl.rcParams['font.family'] = 'Helvetica'
plt.rcParams['axes.linewidth'] = 1.6
plt.rc('xtick', labelsize=tick_fontsize)
plt.rc('ytick', labelsize=tick_fontsize)
plt.rc('axes', labelsize=label_fontsize)
plt.rc('axes', titlesize=title_fontsize)
plt.rc('legend', fontsize=8)    # legend fontsize
plt.rc('figure', titlesize=title_fontsize)  # fontsize of the figure title
plt.rcParams['axes.facecolor'] = 'None'

width = 6.88 + 3
height = width/1.618 

# =============================================================================
# Import all data and fix timestamps to match each other
# =============================================================================

kanl_data = pd.read_csv('kanl_era_data.csv')
kanl_data['date(UTC-2)'] = pd.to_datetime(kanl_data['date(UTC-2)'].values, format = '%d/%m/%Y %H:%M')#.strftime('%d/%m')

print(kanl_data)
rain = kanl_data['PCPT(mm)'].values
t2 = kanl_data['AirTemperature(C)'].values
pressure = kanl_data['AirPressure(hPa)'].values
rh = kanl_data['RelativeHumidity(%)'].values
windspeed = kanl_data['WindSpeed(m/s)'].values
winddir = kanl_data['WindDirection(d)'].values
cloud = kanl_data['CloudCover'].values * 100
time = kanl_data['date(UTC-2)']
print(time)

t2 = windspeed
print('air temp mean, max, min', np.mean(t2), np.max(t2), np.min(t2))


# =============================================================================
#  Creating big graph figure
# =============================================================================
fig = plt.figure()
spec = gridspec.GridSpec(ncols=1, nrows=3, hspace=0.22, height_ratios=[1,1,0.8])

#----------ax1----temperature and rh and pressure-----------------------------------------
ax1 = fig.add_subplot(spec[0])

l1 = ax1.plot(time, t2, color='blue', label='Air Temperature')
ax1.set_ylabel('Air Temperature ($^{o}$C)')
ax1.set_xlim(pd.to_datetime('2022-08-01 00:00:00'), pd.to_datetime('2022-08-16 00:00:00'))

ax1b = ax1.twinx()
l2 = ax1b.plot(time, rh, color='green', label='Relative Humidity')
ax1b.set_ylabel('Relative Humidity (%)')

ax1c = ax1.twinx()
ax1c.spines.right.set_position(("axes", 1.08))
l3 = ax1c.plot(time, pressure, color='k', label='Air Pressure', linestyle='--')
ax1c.set_ylabel('Air Pressure (hPa)')

lns = l1 + l2 + l3
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, facecolor='white', loc='lower right', bbox_to_anchor=(0.59, 0.98), ncol=3)

hours = mdates.DayLocator()
h_fmt = mdates.DateFormatter('%d/%m')
ax1.xaxis.set_major_locator(hours)
ax1.xaxis.set_major_formatter(h_fmt)
ax1.grid(which='major', axis='x', linestyle='--', linewidth=1.0, color='k', zorder=-4)
ax1.text(0.01, 0.85, '(a)', transform=ax1.transAxes, fontsize=20)

ax1.axvspan(xmin=pd.to_datetime('2022-08-03 00:00'), xmax=pd.to_datetime('2022-08-06 00:00'), facecolor='lightblue', alpha=0.3, zorder=-4)
ax1.axvspan(xmin=pd.to_datetime('2022-08-07 00:00'), xmax=pd.to_datetime('2022-08-11 00:00'), facecolor='lightgreen', alpha=0.3, zorder=-4)
ax1.axvspan(xmin=pd.to_datetime('2022-08-11 00:00'), xmax=pd.to_datetime('2022-08-16 00:00'), facecolor='lightgray', alpha=0.4, zorder=-4)

#----------ax2----melt and rain-----------------------------------------
#
ax2 = fig.add_subplot(spec[1])

ax2.bar(time, rain, width=1/24, zorder=-4, color='limegreen', edgecolor='k', label='Rainfall')
ax2.set_ylabel('Rainfall (mm)')
ax2.set_xlim(pd.to_datetime('2022-08-01 00:00:00'), pd.to_datetime('2022-08-16 00:00:00'))
ax2.set_ylim(0, 3)

ax2b = ax2.twinx()
l2 = ax2b.plot(time, cloud, color='k', label='Cloud Cover')
ax2b.set_ylabel('Cloud Cover (%)')

lns = l2
labs = [l.get_label() for l in lns]
ax2b.legend(lns, labs, facecolor='white', loc='lower right', bbox_to_anchor=(0.57, 0.16), ncol=2)
ax2.legend(loc='center', facecolor='white')

hours = mdates.DayLocator()
h_fmt = mdates.DateFormatter('%d/%m')
ax2.xaxis.set_major_locator(hours)
ax2.xaxis.set_major_formatter(h_fmt)
ax2.grid(which='major', axis='x', linestyle='--', linewidth=1.0, color='k', zorder=-4)
ax2.text(0.01, 0.85, '(b)', transform=ax2.transAxes, fontsize=20)

ax2.axvspan(xmin=pd.to_datetime('2022-08-03 00:00'), xmax=pd.to_datetime('2022-08-06 00:00'), facecolor='lightblue', alpha=0.3, zorder=-4)
ax2.axvspan(xmin=pd.to_datetime('2022-08-07 00:00'), xmax=pd.to_datetime('2022-08-11 00:00'), facecolor='lightgreen', alpha=0.3, zorder=-4)
ax2.axvspan(xmin=pd.to_datetime('2022-08-11 00:00'), xmax=pd.to_datetime('2022-08-16 00:00'), facecolor='lightgray', alpha=0.4, zorder=-4)

#----------ax2----melt and rain-----------------------------------------

u = windspeed*np.sin(np.radians(winddir-180))
v = windspeed*np.cos(np.radians(winddir-180))
y = np.full(len(time[::6]), -1)

ax3 = fig.add_subplot(spec[2])

ax3.plot(time, windspeed, color='r', label='Wind Speed')
ax3.barbs(time[::6], y, u[::6],v[::6], label='Wind Direction')
ax3.set_ylim(-3,9)
ax3.set_yticks([0,2,4,6,8])
ax3.set_ylabel('Wind Speed (m/s)')
ax3.set_xlim(pd.to_datetime('2022-08-01 00:00:00'), pd.to_datetime('2022-08-16 00:00:00'))
ax3.legend(facecolor='white')

hours = mdates.DayLocator()
h_fmt = mdates.DateFormatter('%d/%m')
ax3.xaxis.set_major_locator(hours)
ax3.xaxis.set_major_formatter(h_fmt)
ax3.grid(which='major', axis='x', linestyle='--', linewidth=1.0, color='k', zorder=-4)
ax3.text(0.01, 0.85, '(c)', transform=ax3.transAxes, fontsize=20)

ax3.axvspan(xmin=pd.to_datetime('2022-08-03 00:00'), xmax=pd.to_datetime('2022-08-06 00:00'), facecolor='lightblue', alpha=0.3, zorder=-4)
ax3.axvspan(xmin=pd.to_datetime('2022-08-07 00:00'), xmax=pd.to_datetime('2022-08-11 00:00'), facecolor='lightgreen', alpha=0.3, zorder=-4)
ax3.axvspan(xmin=pd.to_datetime('2022-08-11 00:00'), xmax=pd.to_datetime('2022-08-16 00:00'), facecolor='lightgray', alpha=0.4, zorder=-4)



# =============================================================================
# saving figure
# =============================================================================
fig.set_size_inches(width, height)
plt.savefig('met_graph.pdf', bbox_inches = "tight")
plt.show()
