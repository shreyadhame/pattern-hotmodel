#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__title__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
## Load modules
import os
import sys

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import numpy.ma as ma
import xarray as xr
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import scipy.signal
import pymannkendall as mk
import proplot as pplt

#My modules
from load_le_data import * 
from func import *
from plot_map import *

plt.rcParams.update({'hatch.color': '#363636'})

#============================================================
### Execute script

if __name__ == "__main__":
    #Arrange models according to increasing ECS 
    mods = [gfdl, miroc6, miroc, giss, cesm1, nor, mpi, canesm2, access, gfdlcm3, csiro, ipsl, cnrm, cesm2, canesm5]
    ecs = [2.44, 2.6 , 2.66, 2.71, 2.94, 3.03, 3.63, 3.7 , 3.88, 3.95, 4.09, 4.7 , 4.9 ,5.15, 5.64]
    mod_units = ['K', 'C', 'C', 'C', 'C', 'C', 'K', 'K', 'C', 'K', 'K', 'C', 'C', 'K', 'C']

    # mods = np.random.choice(mods,14)

    #convert to degC
    mod_data = mods
    for i in range(len(mods)):
        if mod_units[i] == 'K':
            mod_data[i] = mods[i] - 273.15
        elif mod_units == 'C':
            pass

    #Extract data from kleptp
    db = klepto.archives.dir_archive('mphi',serialized=True,cached=False)
    ew_obs = db['ew_obs']
    ew_mods = db['ew_mods']
    ew_trends = db['ew_trends']
    ew_mtrends = db['ew_mtrends']
    ew_mphi = db['ew_mphi']
    mti = db['mti']

    #Calculate r2 and slope for EffCS and phi
    ew_r = []
    ew_slope = []
    ew_intercept = []
    ew_p = []
    for l in range(ew_mphi.shape[1]): 
        if np.count_nonzero(ew_mphi[:,l])==0:
            ew_r.append(np.nan)
            ew_slope.append(np.nan)
            ew_intercept.append(np.nan)
            ew_p.append(np.nan)
        else:
            #Calculate R2
            r_value = stats.linregress(ew_mphi[:,l],ecs)[2]
            ew_r.append(r_value**2)
            #Calculate slope
            ew_slope.append(stats.linregress(ew_mphi[:,l],ecs)[0])
            #Calculate intercept
            ew_intercept.append(stats.linregress(ew_mphi[:,l],ecs)[1])
            #Calculate significance 
            p = stats.linregress(ew_mphi[:,l],ecs)[3]
            ew_p.append(p)

    shape = int(np.sqrt(len(ew_r)))
    ew_pr = np.reshape(ew_p,(shape,shape)).T #Reshape to 2D matrix and transpose
    ew_dp = pd.DataFrame(ew_pr, columns=[str(v) for v in years])
    ew_dp = ew_dp.set_index(np.arange(start_year,end_year))
    ew_dpma = ma.masked_where(ew_dp > 0.5, ew_dp)

    ew_rr = np.reshape(ew_r,(shape,shape)).T #Reshape to 2D matrix and transpose
    ew_dr = pd.DataFrame(ew_rr*~ew_dpma.mask, columns=[str(v) for v in years])
    ew_dr = ew_dr.set_index(np.arange(start_year,end_year))

    ew_sloper = np.reshape(ew_slope,(shape,shape)).T #Reshape to 2D matrix and transpose
    ew_dslope = pd.DataFrame(ew_sloper*~ew_dpma.mask, columns=[str(v) for v in years])
    ew_dslope = ew_dslope.set_index(np.arange(start_year,end_year))

    ## PLOT 
    fig = plt.figure(figsize=(19,8))
    widths = [1,0.03,0.12,1,0.03]
    heights = [1]
    spec = gridspec.GridSpec(ncols=5, nrows=1, width_ratios=widths,
                            height_ratios=heights,hspace=0.,wspace=0.)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 3])
    ax3 = fig.add_subplot(spec[0, 2])
    ax3.set_visible(False)
    cax1 = fig.add_subplot(spec[0, 1])
    cax2 = fig.add_subplot(spec[0, 4])

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(ew_dr, dtype=bool))
    # Draw the heatmap with the mask and correct aspect ratio
    vmin = 0.
    vmax = 1
    center = 0.
    h1 = sns.heatmap(ew_dr, mask=mask, vmin=vmin, vmax=vmax, cmap=pplt.Colormap('NegPos'), center=center,\
            xticklabels=4, yticklabels=4,
            square=True, linewidths=.5, ax=ax1, cbar_ax=cax1) #\u03A6 - phi
    # cax1.set_ylabel('$^\circ$C/decade', fontsize=14,labelpad=-50)
    cax1.tick_params(labelsize=12)
    h1.set_yticklabels(h1.get_yticklabels(), rotation = 0, fontsize = 12)
    h1.set_xticklabels(h1.get_xticklabels(), rotation = 90, fontsize = 12)
    cax1.yaxis.set_ticks_position('left')

    #Set labels
    ax1.set_xlabel('Start year',fontsize=14)
    ax1.set_ylabel('End year',fontsize=14)

    #Set limits 
    h1.set_xlim(0,71-19)
    h1.set_ylim(19,71)
    h1.invert_yaxis()

    ax1.set_title('(a) Coefficient of determination [R-Squared]',fontsize=16,pad=15)
    # ax1.set_title('(a) Correlation coefficient',fontsize=16,y=-0.18)

    ##SO
    # Draw the heatmap with the mask and correct aspect ratio
    vmin = -3
    vmax = 3
    center = 0
    h2 = sns.heatmap(ew_dslope, vmin=vmin, vmax=vmax, cmap=pplt.Colormap('NegPos'), center=center,
                    xticklabels=4,yticklabels=4,
            square=True, linewidths=.5, ax=ax2, cbar_ax = cax2) #\u03A6 - phi ,'label': '$^\circ$C/decade'
    cax2.tick_params(labelsize=12)
    # cax2.set_ylabel('[K/C]', fontsize=14) #/\u03C3
    #Set limits 
    h2.set_xlim(0,71-19)
    h2.set_ylim(19,71)
    h2.invert_yaxis()
    # h2.yaxis.set_ticks_position('right') 
    # h2.yaxis.set_label_position('right') 
    h2.set_yticklabels(h2.get_yticklabels(), rotation = 0, fontsize = 12)
    h2.set_xticklabels(h2.get_xticklabels(), rotation = 90, fontsize = 12)
    cax2.yaxis.set_ticks_position('left')

    #Set labels
    ax2.set_xlabel('Start year',fontsize=14)
    ax2.set_ylabel('End year',fontsize=14)

    ax2.set_title('(b) Slope',fontsize=16, pad=15)
    # ax2.set_title('(b) Slope',fontsize=16,y=-0.18)

    ax2.set_zorder(-1)

    # ax1.axhline(44.5)
    # ax1.axvline(20.5)

    plt.savefig('SM_Fig4.png', dpi=300, facecolor='white', bbox_inches="tight")