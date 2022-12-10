#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 09:37:24 2022

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

#-----------------------figure user options------------------------------------
tick_fontsize = 10
label_fontsize = 11
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

width = 6.88 
height = width/1.618 + 0.4

# =============================================================================
# IMPORTING DATA
# =============================================================================
abl_data = pd.read_csv('cosipy_ablation_data_cumsum.csv')
abl_data['date'] = pd.to_datetime(abl_data['date'].values, dayfirst=True)
time = abl_data['date']

abl1 = abl_data['abl_1']
abl1_cosipy = abl_data['abl_1_cosipy']

abl2 = abl_data['abl_2']
abl2_cosipy = abl_data['abl_2_cosipy']

abl3 = abl_data['abl_3']
abl3_cosipy = abl_data['abl_3_cosipy']

abl4 = abl_data['abl_4']
abl4_cosipy = abl_data['abl_4_cosipy']

gps = abl_data['GPS']
gps_cosipy = abl_data['GPS_cosipy']


sdate = pd.to_datetime('2022-08-06 19:00')
edate = pd.to_datetime('2022-08-14 12:00')
# =============================================================================
#  Creating big graph figure
# =============================================================================
fig = plt.figure()
spec = gridspec.GridSpec(ncols=1, nrows=5, hspace=0.2)

#----------ax1----t-----------------------------------------
ax1 = fig.add_subplot(spec[0])
ax1.scatter(time, abl1, color='r', zorder=4, label='Ablation Stake')
ax1.plot(time, abl1_cosipy.values, color='k', label='Modelled Sfc Melt')

ax1.set_ylabel('Surface Melt (cm)', y=-1.8)
ax1.set_yticks([0,4,8, 12]) 
ax1.set_ylim(0, 14)
ax1.legend(fontsize=10, loc=0)
ax1.set_xlim(sdate, edate)
ax1.text(0.38, 0.3, 'Stake 1', horizontalalignment='center', 
         verticalalignment='center', transform=ax1.transAxes, fontsize=11)

hour_locator = mdates.HourLocator(interval=24)
ax1.xaxis.set_major_locator(hour_locator)
ax1.xaxis.set_major_formatter(dates.DateFormatter('%d'))
ax1.set_xticklabels([])

#----------ax2----t-----------------------------------------
ax2 = fig.add_subplot(spec[1])
ax2.scatter(time, abl2, color='r', zorder=4)
ax2.plot(time, abl2_cosipy.values, color='k')

ax2.set_xlim(sdate, edate)
ax2.set_yticks([0,4,8, 12]) 
ax2.set_ylim(0, 14)
ax2.text(0.08, 0.12, 'Stake 2', horizontalalignment='center', 
         verticalalignment='center', transform=ax2.transAxes, fontsize=11)

hour_locator = mdates.HourLocator(interval=24)
ax2.xaxis.set_major_locator(hour_locator)
ax2.xaxis.set_major_formatter(dates.DateFormatter('%d'))
ax2.set_xticklabels([])

#----------ax3----t-----------------------------------------
ax3 = fig.add_subplot(spec[2])
ax3.scatter(time, abl3, color='r', zorder=4)
ax3.plot(time, abl3_cosipy.values, color='k')

ax3.set_xlim(sdate, edate)
ax3.set_yticks([0,4,8, 12]) 
ax3.set_ylim(0, 14)
ax3.text(0.08, 0.12, 'Stake 3', horizontalalignment='center', 
         verticalalignment='center', transform=ax3.transAxes, fontsize=11)

hour_locator = mdates.HourLocator(interval=24)
ax3.xaxis.set_major_locator(hour_locator)
ax3.xaxis.set_major_formatter(dates.DateFormatter('%d'))
ax3.set_xticklabels([])

#----------ax4----t-----------------------------------------
ax4 = fig.add_subplot(spec[3])
ax4.scatter(time, abl4, color='r', zorder=4)
ax4.plot(time, abl4_cosipy.values, color='k')

ax4.set_xlim(sdate, edate)
ax4.set_yticks([0,4,8, 12]) 
ax4.set_ylim(0, 14)
ax4.text(0.08, 0.12, 'Stake 4', horizontalalignment='center', 
         verticalalignment='center', transform=ax4.transAxes, fontsize=11)

hour_locator = mdates.HourLocator(interval=24)
ax4.xaxis.set_major_locator(hour_locator)
ax4.xaxis.set_major_formatter(dates.DateFormatter('%d'))
ax4.set_xticklabels([])


#----------GPS----t-----------------------------------------
ax5 = fig.add_subplot(spec[4])
ax5.scatter(time, gps, color='r', zorder=4)
ax5.plot(time, gps_cosipy.values, color='k')

ax5.set_xlim(sdate, edate)
ax5.set_yticks([0,5,10,15,20,25]) 
ax5.set_ylim(0, 25)
ax5.text(0.08, 0.25, 'GPS stake', horizontalalignment='center', 
         verticalalignment='center', transform=ax5.transAxes, fontsize=11)

hour_locator = mdates.HourLocator(interval=24)
ax5.xaxis.set_major_locator(hour_locator)
ax5.xaxis.set_major_formatter(dates.DateFormatter('%d'))
#ax5.set_xticklabels([])
ax5.set_xlabel('August')



fig.set_size_inches(width, height)
plt.savefig('stake_comparison.pdf')
plt.show()


