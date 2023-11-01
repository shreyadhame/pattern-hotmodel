#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__title__ = "Map of observed trend"
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
## Load modules
import os
import sys

import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser
import gc 
import klepto
import pandas as pd
import proplot as pplt
import pymannkendall as mk
import seaborn as sns
import scipy.signal
from scipy.stats import iqr, norm
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import xarray as xr

#My modules
from load_le_data import * 
from func import *
from plot_map import *

plt.rcParams.update({'hatch.color': '#363636'})
pplt.rc['tick.lenratio']=0.02
pplt.rc['savefig.transparent']=True
#============================================================
def func_mkslope(x):
    slope = mk_test(x)[-1]
    return slope

#============================================================
### Execute script

if __name__ == "__main__":
    start_year = '1979'
    end_year = '2020'
    #Select time 
    obs = obs.sel(time=slice(start_year,end_year))

    #Calculate trend 
    obst = np.apply_along_axis(func_mkslope,0, np.nan_to_num(obs))

    # PLOT 
    fig = plt.figure(figsize=(12,7))
    widths = [1]
    heights = [1, 0.05]
    spec = gridspec.GridSpec(ncols=1, nrows=2, width_ratios=widths,
                            height_ratios=heights)
    projection=ccrs.Robinson(180)
    ax1 = fig.add_subplot(spec[0, 0], projection=projection)
    ax_cb = fig.add_subplot(spec[1, :])

    # Set default font size
    plt.rcParams.update({"font.size": "22"})

    axs = [ax1]
    titles=['(a) Observed SST Trend (1979-2020)']

    levels = np.arange(-1.6,1.65,0.05)
    cmap = pplt.Colormap('ColdHot')

    obst[obst == 0] = np.nan

    for i in range(len(axs)):
            plot_maps_gridspec(axs[i],obst*42,lon,lat,levels=levels,mp=0.,
                            cmap=cmap,
                            ticks=False,land=True,title=titles[i],loc_title='center',pad=5)
            
            # #Southern Ocean
            # axs[i].plot([0, 360, 360, 0, 0], [-45, -45, -65, -65, -45],
            #  color='black', linewidth=1, marker='.',
            #  transform=ccrs.PlateCarree())
    
            #Western Pacific Ocean box
            axs[i].plot([110%360, 180%360, 180%360, 110%360, 110%360], [-5, -5, 5, 5, -5],
             color='black', linewidth=1, marker='.',
             transform=ccrs.PlateCarree())

            #Eastern Pacific Ocean box
            axs[i].plot([180%360, -80%360, -80%360, 180%360, 180%360], [-5, -5, 5, 5, -5],
             color='black', linewidth=1, marker='.',
             transform=ccrs.PlateCarree())

    #colorbar
    mp=0.
    vmin=levels[0]
    vmax=np.round(levels[-1],2)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cbar = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation='horizontal',\
                                    extend='both',ticks=levels[::8])
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(label='$^\circ$C',size=20)

    plt.savefig('Fig_1a.png', dpi=300, transparent=True, bbox_inches="tight")`
