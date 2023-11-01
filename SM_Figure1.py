#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__title__ = "Illustration of phi"
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
# Load modules
import os
import sys

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
import xarray as xr

#My modules 
from load_le_data import *
from func import *
from plot_map import *

#============================================================
### Execute script

if __name__ == "__main__":

    start_yr = '1990'
    end_yr = '2015'
    
    #Load COBE/ERSSTv5/HadISST SST data
    path = '/work/mh0033/m300952/OBS/'
    cobe = xr.open_dataset(path+'COBE/cobe_sst_yr_1950-2021_g025.nc').sst
    ersst = xr.open_dataset(path+'ERSST/ersst_sst_yr_1950-2021_g025.nc').sst
    hadsst = xr.open_dataset(path+'HADISST/HadISST_sst_yr_1950-2021_g025.nc')
    lon=hadsst.lon
    lat=hadsst.lat
    hadsst = hadsst.sst
    obs = xr.concat([cobe,ersst,hadsst],dim="y",join='override')
    obs = obs.sel(time=slice(start_yr,end_yr)).squeeze()
    
    path = '/work/mh0033/m300952/CMIP'

    #Select models
    miroc6 = xr.open_mfdataset(path+'/MIROC6/tos/*g025.nc',combine='nested',concat_dim='depth').tos #50 members degC
    miroc6 = miroc6.sel(time=slice(start_yr,end_yr)).squeeze()
    
    canesm5 = xr.open_mfdataset(path+'/CanESM5/tos/*g025.nc',combine='nested',concat_dim='depth').tos #50 members degC
    canesm5 = canesm5.sel(time=slice(start_yr,end_yr)).squeeze()
    lon=canesm5.lon
    lat=canesm5.lat

    mod1 = miroc6
    mod2 = canesm5
    
    #Eastern Pacific (EP)
    lat1 = 5
    lat2 = -5
    lone1 = -180%360
    lone2 = -80%360
    #Western Pacific (WP)
    lonw1 = 110
    lonw2 = 180
    
    #Select EP region 
    obs_reg1, lons, lats = sel_reg(obs,lon,lat,lat1=lat2,lat2=lat1,lon1=lone1,lon2=lone2)
    mod1_reg1,lons,lats = sel_reg(mod1,lon,lat,lat1=lat2,lat2=lat1,lon1=lone1,lon2=lone2)
    mod2_reg1,lons,lats = sel_reg(mod2,lon,lat,lat1=lat2,lat2=lat1,lon1=lone1,lon2=lone2)
    #Take spatial mean
    obs_ts1 = [wgt_mean(v,lons,lats) for v in obs_reg1]
    mod1_ts1=[wgt_mean(ma.masked_where(v<-5,v),lons,lats) for v in mod1_reg1]
    mod2_ts1=[wgt_mean(ma.masked_where(v<-5,v),lons,lats) for v in mod2_reg1]

    #Select WP region
    obs_reg2, lons, lats = sel_reg(obs,lon,lat,lat1=lat2,lat2=lat1,lon1=lonw1,lon2=lonw2)
    mod1_reg2,lons,lats = sel_reg(mod1,lon,lat,lat1=lat2,lat2=lat1,lon1=lonw1,lon2=lonw2)
    mod2_reg2,lons,lats = sel_reg(mod2,lon,lat,lat1=lat2,lat2=lat1,lon1=lonw1,lon2=lonw2)
    #Take spatial mean
    obs_ts2 = [wgt_mean(v,lons,lats) for v in obs_reg2]
    mod1_ts2=[wgt_mean(ma.masked_where(v<-5,v),lons,lats) for v in mod1_reg2]
    mod2_ts2=[wgt_mean(ma.masked_where(v<-5,v),lons,lats) for v in mod2_reg2]

    #Calculate east-west gradient
    ew_obs = np.stack([np.array(obs_ts1[j] - obs_ts2[j]) for j in range(len(obs_ts2))])
    ew_mod1 = np.stack([np.array(mod1_ts1[j] - mod1_ts2[j]) for j in range(len(mod1_ts2))])
    ew_mod2 = np.stack([np.array(mod2_ts1[j] - mod2_ts2[j]) for j in range(len(mod2_ts2))])

    #Calculate trends
    obs_trends = [mk_test(v)[-1]*10 for v in ew_obs]
    mod1_trends = [mk_test(v)[-1]*10 for v in ew_mod1]
    mod2_trends = [mk_test(v)[-1]*10 for v in ew_mod2]

    #Distribution
    mod1_data = (mod1_trends - np.nanmean(mod1_trends))/np.std(mod1_trends)
    obs1_data = (obs_trends - np.nanmean(mod1_trends))/np.std(mod1_trends)
    mod2_data = (mod2_trends - np.nanmean(mod2_trends))/np.std(mod2_trends)
    obs2_data = (obs_trends - np.nanmean(mod2_trends))/np.std(mod2_trends)

    #PLOT 
    fig = plt.figure(figsize=(7,3))
    widths = [1,1]
    heights = [1]
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=widths,
                              height_ratios=heights,hspace=0.,wspace=0.1)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])

    
    x = mod1_data
    num_bins = int(np.round(np.sqrt(len(mod1_data))))
    ax1.hist(x, bins=num_bins, facecolor='#E1BE6A', alpha=0.5)
    ax1.set_xlim(-6,6)
    ax1.set_xticks(np.arange(-6, 7, 1.0))
    for label in ax1.get_xticklabels()[1::2]:
        label.set_visible(False)
    y = np.linspace(-4, 4, 1000)
    bin_width = (x.max() - x.min()) / num_bins
    ax1.plot(y, stats.norm.pdf(y) * len(mod1_data) * bin_width, color = '#aa8222', lw=2)
    ax1.axvline(np.nanmean(mod1_data),color='#d4a22b',linewidth=3)
    ax1.axvline(-2,0,0.1,linestyle='--',color='k')
    ax1.axvline(2,0,0.1,linestyle='--',color='k')
    colors = ['#26648e','#4f8fc0','#53d2dc']
    [ax1.axvline(obs1_data[i],color=colors[i],linewidth=3,alpha=0.5) for i in range(len(obs1_data))]
    
    #set labels 
    ax1.set_xlabel(r'$\sigma$',fontsize=12)
    
    #remove spines 
    ax1.spines[['left', 'right', 'top']].set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set_title('MIROC6')
    

    x = mod2_data
    num_bins = int(np.round(np.sqrt(len(mod2_data))))
    ax2.hist(x, bins=num_bins, facecolor='#E1BE6A', alpha=0.5)
    ax2.set_xlim(-6,6)
    ax2.set_xticks(np.arange(-6, 7, 1.0))
    for label in ax2.get_xticklabels()[1::2]:
        label.set_visible(False)
    y = np.linspace(-4, 4, 1000)
    bin_width = (x.max() - x.min()) / num_bins
    ax2.plot(y, stats.norm.pdf(y) * len(mod2_data) * bin_width, color = '#aa8222', lw=2)
    ax2.axvline(np.nanmean(mod2_data),color='#d4a22b',linewidth=3)
    ax2.axvline(-2,0,0.1,linestyle='--',color='k')
    ax2.axvline(2,0,0.1,linestyle='--',color='k')
    colors = ['#26648e','#4f8fc0','#53d2dc']
    [ax2.axvline(obs2_data[i],color=colors[i],linewidth=3,alpha=0.5) for i in range(len(obs2_data))]
    
    #set labels 
    ax2.set_xlabel(r'$\sigma$',fontsize=12)
    
    #remove spines 
    ax2.spines[['left', 'right', 'top']].set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    ax2.set_title('CanESM5')
    #Legend 
    cs = ['#26648e','#4f8fc0','#53d2dc','#d4a22b']
    line = [Line2D([0], [0], color=c, linewidth=3) for c in cs]
    label = ['COBE','ERSSTv5','HadISST','Ensemble \n mean']
    # add manual symbols to auto legend
    ax2.legend(line,label,frameon=False, bbox_to_anchor=(0.6,0.5))

    #title
    plt.suptitle('Trends in E-W Pacific SST gradient (1990-2015)',fontsize=12,y=1.05)

    #save figure
    plt.savefig('SM_Fig1.png',dpi=300,transparent=False,bbox_inches='tight')
