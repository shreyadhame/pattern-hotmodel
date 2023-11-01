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

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np
import numpy.ma as ma
import pandas as pd
import pymannkendall as mk
import proplot as pplt
import seaborn as sns
import scipy.signal
from scipy import stats
from sklearn.linear_model import LinearRegression
import xarray as xr

#My modules
from load_le_data import * 
from func import *
from plot_map import *

plt.rcParams.update({'hatch.color': '#363636'})
pplt.rc['tick.lenratio']=0.02
pplt.rc['savefig.transparent']=True

#============================================================
### Execute script

if __name__ == "__main__":
    
    #Extract data from klepto
    db = klepto.archives.dir_archive('mphi',serialized=True,cached=False)
    ew_obs = db['ew_obs']
    ew_mods = db['ew_mods']
    ew_trends = db['ew_trends']
    ew_mtrends = db['ew_mtrends']
    ew_mtrends_all = db['ew_mtrends_all']
    ew_mtrendsr = db['ew_mtrendsr']
    ew_mphi = db['ew_mphi']
    mti = db['mti']
    
    ew_mtrends = np.stack(ew_mtrends)
    #Reshape all trends 
    n_em = [v.shape[0] for v in ew_mods]
    ew_mtrends_allr = [np.reshape(ew_mtrends_all[i],(n_em[i],5041)) for i in range(len(n_em))]
    
    #Calculate r2 and slope for E-W (model mean trend) and random model trend - 100 times
    
    ew_mr = [[] for _ in range(100)]
    ew_mslope = [[] for _ in range(100)]
    for k in range(100):
        rand = []
        for i in range(len(ew_mods)):
            rand.append(np.apply_along_axis(np.random.choice,0,ew_mtrends_allr[i],size=1))
        rand = np.nan_to_num(np.stack(rand).squeeze())
        for l in range(ew_mtrends.shape[1]):
            if np.count_nonzero(rand[:,l])==0:
                ew_mr[k].append(np.nan)
                ew_mslope[k].append(np.nan)
            else:
                y = rand[:,l]
                x = ew_mtrends[:,l]
                lr = LinearRegression()
                lr.fit(x.reshape(-1, 1), y)
                ew_mr[k].append(lr.score(x.reshape(-1, 1),y))
                #Calculate slope
                ew_mslope[k].append(slope)

    ew_mslope = np.stack(ew_mslope)
    ew_mr = np.stack(ew_mr)
    
    #Take mean of all 100 draws 
    ew_mrm = np.nanmean(ew_mr, axis=0)
    ew_mslopem = np.nanmean(ew_mslope, axis=0)

    shape = int(np.sqrt(len(ew_trends)))
    ew_mrr = np.reshape(ew_mr,(shape,shape)).T #Reshape to 2D matrix and transpose
    ew_mdr = pd.DataFrame(ew_mrr, columns=[str(v) for v in years])
    ew_mdr = ew_mdr.set_index(np.arange(start_year,end_year))

    ew_msloper = np.reshape(ew_mslope,(shape,shape)).T #Reshape to 2D matrix and transpose
    ew_mdslope = pd.DataFrame(ew_msloper, columns=[str(v) for v in years])
    ew_mdslope = ew_mdslope.set_index(np.arange(start_year,end_year))

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
    mask = np.triu(np.ones_like(ew_mrr, dtype=bool))
    # Draw the heatmap with the mask and correct aspect ratio
    vmin = 0
    vmax = 1
    center = 0.
    h1 = sns.heatmap(ew_mdr, mask=mask, vmin=vmin, vmax=vmax, cmap=pplt.Colormap('NegPos'), center=center,\
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
    
    # Draw the heatmap with the mask and correct aspect ratio
    vmin = 0
    vmax = 1.5
    center = 0
    h2 = sns.heatmap(ew_mdslope, vmin=vmin, vmax=vmax, cmap=pplt.Colormap('NegPos'), center=center,
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
    
    plt.savefig('SM_Fig5.png', dpi=300, facecolor='white', bbox_inches="tight")
