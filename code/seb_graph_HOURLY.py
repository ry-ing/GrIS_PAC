#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 17:19:39 2022

@author: ryaning
"""

import numpy as np
import pandas as pd
import matplotlib
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

width = 6.88 + 8
height = width/1.618 

# =============================================================================
# Import all data and fix timestamps to match each other
# =============================================================================

cosipy_data = pd.read_csv('catchment_data_mean_v4.csv')
# cosipy_data['date'] = pd.PeriodIndex(cosipy_data['date'], freq='H')
# cosipy_data['date'] = cosipy_data['date'].astype(str)
cosipy_data['date'] = pd.to_datetime(cosipy_data['date'].values, format = '%Y-%m-%d').strftime('%d %H:%Mh')


sdate = '03'
edate = '14'

SW_in = cosipy_data['G'].values
SW_net = SW_in * (1-0.57)
LW_net = cosipy_data['LWin'].values + cosipy_data['LWout'].values
H = cosipy_data['H'].values #SENSIBLE
LE = cosipy_data['LE'].values #LATENT
B = cosipy_data['B'].values #GROUND
QRR = cosipy_data['QRR'].values #RAIN
ME = cosipy_data['ME'].values
#me2 = SW_net + LW_net + H + LE + B + QRR
SW_net = ME - LW_net - H - LE - B - QRR

sub = cosipy_data['SUBLIMATION'].values
evap = cosipy_data['EVAPORATION'].values
dep = cosipy_data['DEPOSITION'].values
cond = cosipy_data['CONDENSATION'].values

time = cosipy_data['date']#.values

t2 = cosipy_data['T2'].values
RAIN = cosipy_data['RAIN'].values
surfm = cosipy_data['surfM'].values
cloud = cosipy_data['N'].values * 100
u2 = cosipy_data['U2'].values
RH = cosipy_data['RH2'].values

df = pd.DataFrame(index=time, 
                  data={'LW net': LW_net, 
                        'SW net' : SW_net,
                        'Sensible Heat Flux': H, 
                        'Latent Heat Flux': LE, 
                        'Ground Heat Flux': B,
                        'Rain Heat Flux' : QRR})

df_LE = pd.DataFrame(index=time, 
                  data={'Sublimation': sub, 
                        'Evaporation' : evap,
                        'Deposition': dep, 
                        'Condensation': cond})

print(df_LE.index, df.index)

# =============================================================================
#  Creating big graph figure
# =============================================================================
fig = plt.figure()
spec = gridspec.GridSpec(ncols=1, nrows=3, hspace=0.21, height_ratios=[3,1,2])

#----------ax1----temperature and SWin-----------------------------------------
ax1 = fig.add_subplot(spec[0])

df.plot(kind="bar", stacked=True, width=1, edgecolor='k', linewidth=0.05, colormap='flag', ax=ax1, rot=0, use_index=True)
lns2 = ax1.plot(time, ME, color='limegreen', linestyle='--', linewidth=1.3, label='Available Melt Energy')
ax1.set_ylabel('Available Melt Energy (W m$^{-2}$)')
ax1.set_ylim(-150,450)

ax1b = ax1.twinx()
lns1 = ax1b.plot(time, t2, color='k', linestyle='--', linewidth=2,  label='Air Temperature')
ax1b.set_ylabel('Air Temperature ($^{o}$C)')
ax1b.set_ylim(-4,8)

ax1c = ax1.twinx()
ax1c.spines.right.set_position(("axes", 1.05))
lns2 = ax1c.plot(time, u2, color='g', linestyle='solid', linewidth=1,  label='Wind Speed')
ax1c.set_ylabel('Wind Speed (m/s)', y=0.15)
ax1c.yaxis.set_ticks(np.arange(0, 9, 2))
ax1c.set_ylim(0, 38)

ax1.set_xlabel('August', color='w')
ax1.axhline(y=0, color='grey', linestyle='--')
ax1.text(0.01, 0.91, '(a)', transform=ax1.transAxes, fontsize=25)

lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1b.legend(lns, labs, facecolor='white', loc='upper left', bbox_to_anchor=(0.65, 1.15), ncol=2)
ax1.legend(ncol=3, framealpha=0.9, facecolor='white', loc='upper left', bbox_to_anchor=(0.05, 1.3))


major_ticks = ['01 00:00h', '02 00:00h', '03 00:00h', '04 00:00h', '05 00:00h', '06 00:00h', '07 00:00h', '08 00:00h', '09 00:00h', '10 00:00h', '11 00:00h', '12 00:00h', '13 00:00h', '14 00:00h', '15 00:00h', '16 00:00h', '17 00:00h']
ax1.set_xticks(major_ticks)
ax1.set_xlim('01 22:00h', '17 00:00h')
ax1.grid(which='major', axis='x', linestyle='--', linewidth=1.0, color='k', zorder=-4)

ax1.axvspan(xmin='03 00:00h', xmax='06 00:00h', facecolor='lightblue', alpha=0.3, zorder=-4)
ax1.axvspan(xmin='07 00:00h', xmax='11 00:00h', facecolor='lightgreen', alpha=0.3, zorder=-4)
ax1.axvspan(xmin='11 00:00h', xmax='16 00:00h', facecolor='lightgray', alpha=0.4, zorder=-4)

#-----------------------------------------------
ax3 = fig.add_subplot(spec[1])

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["dodgerblue","navy","red",'orange' ])
df_LE.plot(kind="bar", stacked=True, width=1, edgecolor='k', linewidth=0.05, colormap=cmap, ax=ax3, rot=0, use_index=True)
ax3b = ax3.twinx()
ax3b.plot(time, RH, c='k')
ax3b.set_ylabel('RH (%)')
ax3.set_ylabel('(m w.e.)')
ax3.legend(loc='upper left', bbox_to_anchor=(0.17, 0.09), ncol=4)

major_ticks = ['01 00:00h', '02 00:00h', '03 00:00h', '04 00:00h', '05 00:00h', '06 00:00h', '07 00:00h', '08 00:00h', '09 00:00h', '10 00:00h', '11 00:00h', '12 00:00h', '13 00:00h', '14 00:00h', '15 00:00h', '16 00:00h', '17 00:00h']
ax3.set_xticks(major_ticks)
ax3.set_xlim('01 22:00h', '17 00:00h')
ax3.grid(which='major', axis='x', linestyle='--', linewidth=1.0, color='k', zorder=-4)
ax3.set_xticklabels([])
ax3.set_xlabel('August', color='w')
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax3.text(0.01, 0.75, '(b)', transform=ax3.transAxes, fontsize=25)


ax3.axvspan(xmin='03 00:00h', xmax='06 00:00h', facecolor='lightblue', alpha=0.3, zorder=-4)
ax3.axvspan(xmin='07 00:00h', xmax='11 00:00h', facecolor='lightgreen', alpha=0.3, zorder=-4)
ax3.axvspan(xmin='11 00:00h', xmax='16 00:00h', facecolor='lightgray', alpha=0.4, zorder=-4)


#----------ax2----melt and rain-----------------------------------------
ax2 = fig.add_subplot(spec[2])

df = pd.DataFrame(index=time, 
                  data={'Rainfall': RAIN,
                        'surfM': surfm})

df['Rainfall'].plot(kind='bar', width=1, color='limegreen', edgecolor='k', rot=0, use_index=True)#, label='Rainfall', color='green', linestyle='--')
ax2.set_ylabel('Rainfall (mm)')
ax2.set_ylim(0,2.5)
#ax2.set_xlim(0.5, 14.5)
ax2.set_xlabel('August')

ax2b = ax2.twinx()
lns1 = ax2b.plot(time, surfm, color='r', linestyle='--', label='Surface Melt', linewidth=2)
ax2b.set_ylabel('Surface Melt (m w.e.)')

ax2c = ax2.twinx()
ax2c.spines.right.set_position(("axes", 1.04))
lns3 = ax2c.plot(time, cloud, color='grey', linestyle='solid', label='Cloud Cover', linewidth=1)
ax2c.set_ylabel('Cloud Cover (%)')

ax2.grid(which='major', axis='x', linestyle='--', linewidth=1.0, color='k', zorder=-4)
ax2.legend(loc='upper right', bbox_to_anchor=(0.98, -0.1))
lns = lns1 + lns3
labs = [l.get_label() for l in lns]
ax2b.legend(lns, labs, facecolor='white', loc='lower right', bbox_to_anchor=(0.88, -0.32), ncol=2)
ax2.text(0.01, 0.85, '(c)', transform=ax2.transAxes, fontsize=25)

major_ticks = ['01 00:00h', '02 00:00h', '03 00:00h', '04 00:00h', '05 00:00h', '06 00:00h', '07 00:00h', '08 00:00h', '09 00:00h', '10 00:00h', '11 00:00h', '12 00:00h', '13 00:00h', '14 00:00h', '15 00:00h', '16 00:00h', '17 00:00h']
ax2.set_xticks(major_ticks)
ax2.set_xlim('01 22:00h', '17 00:00h')
ax2.grid(which='major', axis='x', linestyle='--', linewidth=1.0, color='k', zorder=-4)

ax2.axvspan(xmin='03 00:00h', xmax='06 00:00h', facecolor='lightblue', alpha=0.3, zorder=-4)
ax2.axvspan(xmin='07 00:00h', xmax='11 00:00h', facecolor='lightgreen', alpha=0.3, zorder=-4)
ax2.axvspan(xmin='11 00:00h', xmax='16 00:00h', facecolor='lightgray', alpha=0.4, zorder=-4)



# =============================================================================
# saving figure
# =============================================================================
fig.set_size_inches(width, height)
plt.savefig('seb2.pdf')
plt.show()
