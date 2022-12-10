#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to calculate KAN_L PROMICE ice velocities at different periods

Saves ice velocities to CSV file
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import dates
import matplotlib as mpl
import math
import sys

# Need to download this from GitHub: https://github.com/nsidc/polarstereo-lonlat-convert-py.git
# 1) go to URL, click on "code", "download zip"
# 2) move downloaded folder to sensible location, and unzip
# 3) openup powershell on PC, navigate to where folder is
# 4) type: " pip install --editable /path/to/cloned/polarstereo-lonlat-convert-py-main"
from polar_convert.constants import NORTH
from polar_convert import polar_lonlat_to_xy

#---------------MATPLOTLIB USER OPTIONS---------
tick_fontsize = 9
label_fontsize = 12
title_fontsize = 13
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.6
plt.rc('xtick', labelsize=tick_fontsize)
plt.rc('ytick', labelsize=tick_fontsize)
plt.rc('axes', labelsize=label_fontsize)
plt.rc('axes', titlesize=title_fontsize)
plt.rc('legend', fontsize=label_fontsize)    # legend fontsize
plt.rc('figure', titlesize=title_fontsize)  # fontsize of the figure title
plt.rcParams['axes.facecolor'] = 'None'

width = 6.88 +5
height = width/1.618  - 3
#-------------------------------------------------

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

#---Function to calculate ice velocities (scroll past this to use function)------------
def calculate_ice_velocity(MET_DATA, s1_data, start_date, end_date, period, rolling_mean, save_csv, save_fig, plot_s1):
    
    test_date1 = pd.to_datetime(start_date)
    test_date2 = pd.to_datetime(end_date)
    
    if MET_DATA['date(UTC-2)'].iloc[0] >= test_date1:
        print('ERROR! Start date selected is before available dates')
        print('Selected start date:', start_date)
        print('Data start date:', MET_DATA['date(UTC-2)'].iloc[0])
        sys.exit()
        
    if MET_DATA['date(UTC-2)'].iloc[-1] <= test_date2:
        print('ERROR! End date selected is after available dates')
        print('Selected end date:', end_date)
        print('Data end date:', MET_DATA['date(UTC-2)'].iloc[-1])
        sys.exit()
      
    #--------MASK TIME TO DURATION OF FIELDWORK-------------------
    print('\nSelecting data from', start_date, 'to', end_date, '\n')
    mask = (MET_DATA['date(UTC-2)'] >= start_date) & (MET_DATA['date(UTC-2)'] <= end_date)
    MET_DATA = MET_DATA.loc[mask]

    airtemp_sub = MET_DATA
    MET_DATA = MET_DATA[MET_DATA['HorDilOfPrecGPS'] !=1] #Remove timestamps with a high horizontal dilution
    MET_DATA = MET_DATA[MET_DATA['TimeGPS(hhmmssUTC)'] !=-999.0] # remove timestamps with no GPS measurements
    MET_DATA = MET_DATA[MET_DATA['TimeGPS(hhmmssUTC)'] !=1]
    MET_DATA['gps_time'] = pd.to_datetime(MET_DATA['TimeGPS(hhmmssUTC)'], format = '%H%M%S')

    lat = MET_DATA['LatitudeGPS(degN)'].values
    lon = MET_DATA['LongitudeGPS(degW)'].values
    time = MET_DATA['date(UTC-2)'].values
    
    if plot_s1=='True':
        #-----MASKING SENTINEL DATA-----------------------------------
        time_mask = (s1_data['Datenum'] >= start_date) & (s1_data['Datenum'] <= end_date)
        s1_data = s1_data.loc[time_mask]
        s1_data['Smoothed S'] = s1_data['Smoothed S'] / 365
        s1_time = s1_data['Datenum'].values
        s1_speed = s1_data['Smoothed S'].values
        print(s1_data)
    
    if period % 6 ==0:      
        period_days = period/24
        print('\nCalculating ice velocity (m/day) every %s hours or %s days\n' % (period, period_days))
        
        #time from GPS is every 6 hours
        index_no = int(period/6)
        
        x = np.zeros(len(lat))
        y = np.zeros(len(lat))       
        d = np.zeros(len(lat))
        ice_velocity = np.zeros(len(lat)-index_no)
        time_between_array = []
                                 
        true_scale_lat = 70  # true-scale latitude in degrees
        re = 6378.137  # earth radius in km
        e = 0.01671 # earth eccentricity
        hemisphere = NORTH
        
        for i in range(0,len(lat)):
            x[i], y[i] = polar_lonlat_to_xy(lon[i], lat[i], true_scale_lat, re, e, hemisphere) #calculate distances between points
            
            if i>=index_no:
                skip = index_no

                x_diff = (x[i] - x[i-skip]) * 1000 #convert km to metres
                y_diff = (y[i] - y[i-skip]) * 1000 #convert km to metres
                d[i] = math.sqrt( ((x_diff)**2) + ((y_diff)**2) )
                time_diff = (np.datetime64(time[i]) - np.datetime64(time[i-skip])).astype(int) * 1e-9
                #print(time_diff)
                velocity_ms = d[i] / time_diff #velocity in m/s
                ice_velocity[i-index_no] = velocity_ms * 86400 #velocity in metres per day
                
                #-----calculating time between 2 timestamps-----------
                time1 = pd.to_datetime(np.datetime64(time[i-skip]))
                time2 = pd.to_datetime(np.datetime64(time[i]))     
                time_between = time1 + (time2 - time1)/2
                time_between_array.append(time_between)
                

        time_between_array = pd.to_datetime(time_between_array).values
        ice_velocity = moving_average(ice_velocity, rolling_mean)
        
        sliced_airtemp = airtemp_sub[airtemp_sub['date(UTC-2)'].isin(time_between_array)]
        airtemp = sliced_airtemp['AirTemperature(C)']   
        #airtemp = moving_average(airtemp, rolling_mean)

        fig, ax = plt.subplots(figsize=(20,10))
        ln1 = ax.plot(time_between_array, ice_velocity, color='k', label='KAN_L speed')
        ax.scatter(time_between_array, ice_velocity, color='k', s=50, marker='x')
        ax.set_ylabel('Ice Velocity (m day$^{-1}$)')
        ax.grid()
        
        ln2 = ax.plot(s1_time, s1_speed, color='b', label='Sentinel-1 speed')
        ax.scatter(s1_time, s1_speed, color='b', s=50, marker='x')
        ax.set_ylabel('Speed (m day$^{-1}$)')
        
        axb = ax.twinx()
        ln3 = axb.plot(time_between_array, airtemp, color='red', label='KAN_L air temperature')
        axb.set_ylabel('Air Temperature ($^{o}$C)', color='red')
        
        lns = ln1+ln2+ln3
        labs = [l.get_label() for l in lns]
        legend = ax.legend(lns, labs, facecolor='white', bbox_to_anchor=(0.4, 1.0), loc='upper left')
        for line in legend.get_lines():
            line.set_linewidth(4.0)
            
        ax.axvspan(xmin=pd.to_datetime('2022-08-03 00:00'), xmax=pd.to_datetime('2022-08-06 00:00'), facecolor='lightblue', alpha=0.4, zorder=-4)
        ax.axvspan(xmin=pd.to_datetime('2022-08-07 00:00'), xmax=pd.to_datetime('2022-08-11 00:00'), facecolor='lightgreen', alpha=0.4, zorder=-4)
        ax.axvspan(xmin=pd.to_datetime('2022-08-11 00:00'), xmax=pd.to_datetime('2022-08-16 00:00'), facecolor='lightgray', alpha=0.5, zorder=-4)



        # rolling_window = rolling_mean * 6
        # textstr = 'Rolling Mean Window = %.0f hours' % rolling_window
        # ax.text(0.05, 0.9, textstr, transform=ax.transAxes, fontsize=18)

        hour_locator = mdates.HourLocator(interval=period)
        ax.xaxis.set_major_locator(hour_locator)
        ax.xaxis.set_major_formatter(dates.DateFormatter('%d/%m %H:%M'))
        
        if len(lat)<120:
            hour_locator_minor = mdates.HourLocator(interval=6)
            ax.xaxis.set_minor_locator(hour_locator_minor)
            ax.xaxis.set_minor_formatter(dates.DateFormatter('%H:%M'))    
        else:
            #plt.minorticks_off()
            ax.xaxis.set_tick_params(which='minor',bottom=False)
            
        ax.tick_params(axis="x", which="both", rotation=90)
        date_range_title = "%s to %s" % (pd.to_datetime(time_between_array[0]), pd.to_datetime(time_between_array[-1]))
        ax.set_title('KAN_L Ice Velocity (%s hourly or %.0f daily frequency) \n%s Local Time' % (period, period_days, date_range_title))
        fig.set_size_inches(width, height)

        if save_fig=='True':
            date_range_str = "%s_%s" % (pd.to_datetime(time_between_array[0]).strftime('%Y-%m-%d'), pd.to_datetime(time_between_array[-1]).strftime('%Y-%m-%d'))
            filename = 'KAN_L_speed_%s.png' % date_range_str
            plt.savefig(filename, dpi=300, bbox_inches = "tight")
            print('Figure saved to:', filename)
            
        
        plt.show()
        
        if save_csv=='True':       
            velo_name = 'ice_velocity_%.0fh'  %  period
            data = {'date(UTC-2)':time_between_array, velo_name: ice_velocity }
            dataframe = pd.DataFrame(data=data)
            
            firstdate = pd.to_datetime(start_date).strftime('%d%m%y')
            lastdate = pd.to_datetime(end_date).strftime('%d%m%y')
    
            date_file = '%s-%s' % (firstdate, lastdate)
            outfile = 'KAN_L_ice_velocity_%.0fh_%s.csv' % (period, date_file)
            dataframe.to_csv(outfile)
            print(dataframe)
            print('\nIce Velocity CSV saved to:', outfile)
            
        print('mean ice velocity', np.mean(ice_velocity))
    else:
        print('ERROR: Period must be a multiple of 6 hours!!')



#-----------------------------------------------------------------------------------------

# =============================================================================
# IMPORT DATA
# =============================================================================
met_filename = 'KAN_L_2022_localtime.csv' #file directory to edited KAN_L file
MET_DATA = pd.read_csv(met_filename, index_col=0)
MET_DATA['date(UTC-2)'] = pd.to_datetime(MET_DATA['date(UTC-2)'].values)

s1_data = pd.read_csv('ROI001_smoothed_tseries.txt')
s1_data['Datenum'] = pd.to_datetime(s1_data['Datenum'].values)


# =============================================================================
#  USER OPTIONS
# =============================================================================
start_date = '2022-07-31 09:00:00' #date range wanted in local kangerluusauq time
end_date = '2022-08-17 15:00:00'

period = 24 #period of measurements in hours, must be a multiple of 6 hours
rolling_mean = 4 #window of rolling mean average
save_csv = 'false'
save_fig = 'True'
plot_s1 = 'True'

calculate_ice_velocity(MET_DATA, s1_data, start_date, end_date, period, rolling_mean, save_csv, save_fig, plot_s1)













