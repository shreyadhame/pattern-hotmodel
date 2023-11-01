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
import numpy as np
import numpy.ma as ma
import xarray as xr
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

import scipy.signal
import pymannkendall as mk
import proplot as pplt

#My modules
from load_le_data import * 
from func import *
from plot_map import *

plt.rcParams.update({'hatch.color': '#363636'})

#============================================================
def plot_scatter(y1,y2,text=''):
    ticks = np.arange(0.5,71.5)
    years=np.arange(start_year,end_year)
    x = ticks[np.where(years==int(y1))]
    y = ticks[np.where(years==int(y2))]
    ax1.text(x,y,text,horizontalalignment='center',verticalalignment='center',fontsize=15,weight='bold')

#============================================================
### Execute script

if __name__ == "__main__":

    #Extract data from klepto
    db = klepto.archives.dir_archive('mphi',serialized=True,cached=False)
    ew_obs = db['ew_obs']
    ew_mods = db['ew_mods']
    ew_trends = db['ew_trends']
    ew_mtrends = db['ew_mtrends']
    ew_mphi = db['ew_mphi']
    mti = db['mti']

    #Model agreement contours 
    ew_c50 = []
    for l in range(ew_mphi.shape[1]):
        if sum(abs(i) > 2 for i in ew_mphi[:,l]) >= int(50*len(mod_data)/100):
            ew_c50.append(1)
        else:
            ew_c50.append(0)

    ew_c80 = []
    for l in range(ew_mphi.shape[1]):
        if int(sum(abs(i) > 2 for i in ew_mphi[:,l])) >= int(80*len(mod_data)/100):
            ew_c80.append(1)
        else:
            ew_c80.append(0)

    #Create dataframes
    shape = int(np.sqrt(len(ew_trends)))
    ew_trendsr = np.reshape(ew_trends,(shape,shape)).T #Reshape to 2D matrix and transpose
    ew_dft = pd.DataFrame(ew_trendsr, columns=[str(v) for v in years])
    ew_mphimr = np.reshape(np.nanmean(ew_mphi,axis=0),(shape,shape)).T
    ew_mphim = pd.DataFrame(ew_mphimr, columns=[str(v) for v in years])

    ew_c50r = np.reshape(ew_c50,(shape,shape)).T #Reshape to 2D matrix and transpose
    ew_dc50 = pd.DataFrame(ew_c50r, columns=[str(v) for v in years])

    ew_c80r = np.reshape(ew_c80,(shape,shape)).T #Reshape to 2D matrix and transpose
    ew_dc80 = pd.DataFrame(ew_c80r, columns=[str(v) for v in years])

    #Change index values to years
    ew_dft = ew_dft.set_index(np.arange(start_year,end_year))
    ew_mphim = ew_mphim.set_index(np.arange(start_year,end_year))
    ew_dc50 = ew_dc50.set_index(np.arange(start_year,end_year))
    ew_dc80 = ew_dc80.set_index(np.arange(start_year,end_year))

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
    mask = np.triu(np.ones_like(ew_trendsr, dtype=bool))
    # Draw the heatmap with the mask and correct aspect ratio
    vmin = -0.4
    vmax = 0.4
    center= 0
    h1 = sns.heatmap(ew_dft, mask=mask, vmin=vmin, vmax=vmax, cmap=pplt.Colormap('NegPos'), center=center,\
            xticklabels=4, yticklabels=4, alpha=0.8,
            square=True, linewidths=.5, ax=ax1, cbar_ax=cax1) #\u03A6 - phi
    cax1.set_ylabel('SST trend [$^\circ$C decade$^{-1}$]', fontsize=14,labelpad=-75)
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
    # h1.xaxis.set_ticks_position('top') 
    # h1.xaxis.set_label_position('top') 
    
    # # # # #Smooth contours
    # # import scipy.ndimage as ndimage
    # # pw=1.007
    # # pw=1.0
    # # ew_dc50s = ndimage.zoom(ew_dc50, pw)
    # # ew_dc80s = ndimage.zoom(ew_dc80, pw)
    # # from skimage.transform import resize
    # # ew_dc50s= resize(ew_dc50s,(72,72),order=1) 
    # # ew_dc80s= resize(ew_dc80s,(72,72),order=1) 
    
    #Overlay phi contours
    levels=[1]
    c1 = h1.contour(np.arange(.5, ew_dc50.shape[1]), np.arange(.5, ew_dc50.shape[0]), ew_dc50, \
                    levels=levels, interpolation='gaussian', colors='#5D3A9B',linewidth=1.)
    c2 = h1.contour(np.arange(.5, ew_dc80.shape[1]), np.arange(.5, ew_dc80.shape[0]), ew_dc80, \
                    levels=levels, interpolation='nearest', colors='#D55E00',linewidth=1.)
    
    #Add contour legend
    l50 = Line2D([0], [0], label='50%', color='#5D3A9B')
    l80 = Line2D([0], [0], label='80%', color='#D55E00')
    
    leg = ax1.legend(handles=[l50, l80],ncol=2,bbox_to_anchor=[0.95, 0.95],fontsize=16,
                     title='% of models for which the \nobservations fall outside \u00B1 2\u03C3',frameon=False)
    plt.setp(leg.get_title(),fontsize=14)

    #Previous studies
    #Olonscheck 2020
    plot_scatter(1975,2004,text='A')
    plot_scatter(1995,2014,text='A')
    # #Seager et al. 2019
    plot_scatter(1958,2008,text='C')
    plot_scatter(1958,2017,text='C')
    #Watanabe 2021
    plot_scatter(1951,1990,text='B')
    plot_scatter(1951,2010,text='B')
    #Wills 2022
    plot_scatter(1979,2020,text='D')
    
    ax1.set_title('(a) East-West Pacific gradient',fontsize=16,pad=15)
    
    # Draw the heatmap with the mask and correct aspect ratio
    vmin = -4
    vmax = 4
    center = 0
    h2 = sns.heatmap(ew_mphim, vmin=vmin, vmax=vmax, cmap=pplt.Colormap('NegPos'), center=center,
                     xticklabels=4,yticklabels=4, square=True, linewidths=.5, ax=ax2, cbar_ax = cax2) #\u03A6 - phi ,'label': '$^\circ$C/decade'
    # cax2.set_ylabel('\u03A6 [\u03C3]', fontsize=14, labelpad=-75)
    cax2.tick_params(labelsize=12)
    #Set limits 
    h2.set_xlim(0,71-19)
    h2.set_ylim(19,71)
    # h2.xaxis.set_ticks_position('top')
    # h2.xaxis.set_label_position('top') 
    # h2.yaxis.set_ticks_position('right') 
    # h2.yaxis.set_label_position('right') 
    h2.set_yticklabels(h2.get_yticklabels(), rotation = 0, fontsize = 12)
    h2.set_xticklabels(h2.get_xticklabels(), rotation = 90, fontsize = 12)
    cax2.yaxis.set_ticks_position('left')
    ax2.invert_yaxis()
    
    #Set labels
    ax2.set_xlabel('Start year',fontsize=14)
    ax2.set_ylabel('End year',fontsize=14)
    ax2.set_title('(b) Multimodel mean of \u03A6',fontsize=16, pad=15)
    
    # #Smooth contours
    # import scipy.ndimage as ndimage
    # pw=1.007
    # pw=1.0
    # so_dc50s = ndimage.zoom(so_dc50, pw)
    # so_dc80s = ndimage.zoom(so_dc80, pw)
    # from skimage.transform import resize
    # so_dc50s= resize(so_dc50s,(72,72),order=1) 
    # so_dc80s= resize(so_dc80s,(72,72),order=1) 
    
    # #Overlay phi contours
    # levels=[1]
    # c1 = h2.contour(np.arange(.5, so_dc50.shape[1]), np.arange(.5, so_dc50.shape[0]), so_dc50, \
    #                 levels=levels, interpolation='gaussian', colors='#5D3A9B',linewidth=1.)
    # c2 = h2.contour(np.arange(.5, so_dc80.shape[1]), np.arange(.5, so_dc80.shape[0]), so_dc80, \
    #                 levels=levels, interpolation='nearest', colors='#D55E00',linewidth=1.)
    
    h1.add_patch(
         patches.Rectangle(
             (1, 41),
             1.0,
             19.0,
             edgecolor='k',
             fill=False,
             lw=1
         ) )
    
    h1.add_patch(
         patches.Rectangle(
             (8, 59),
             1.0,
             8.0,
             edgecolor='k',
             fill=False,
             lw=1
         ) )
    
    #Add legend 
    ax1.annotate(r"$\bf{" + "A" + "}$" + " Olonscheck et al. 2020", xy=(0.52,0.75),xycoords='axes fraction',fontsize=14, 
                 transform_rotates_text=True)
    ax1.annotate(r"$\bf{" + "B" + "}$" + " Watanabe et al. 2021", xy=(0.52,0.7),xycoords='axes fraction',fontsize=14,
                 transform_rotates_text=True)
    ax1.annotate(r"$\bf{" + "C" + "}$" + " Seager et al. 2019", xy=(0.52,0.65),xycoords='axes fraction',fontsize=14, 
                transform_rotates_text=True)
    ax1.annotate(r"$\bf{" + "D" + "}$" + " Wills et al. 2022", xy=(0.52,0.6),xycoords='axes fraction',fontsize=14, 
                 transform_rotates_text=True)
    ax2.set_zorder(-1)
    
    # axin = ax2.inset_axes([0.6, 0.6, 0.4, 0.4])
    mod_names = ['GFDL-ESM2M', 'MIROC6', 'MIROC-ES2L', 'GISS-E2.1-G', 'CESM1', 'NorCPM1', 'MPI-ESM', 'CanESM2', 'ACCESS-ESM1.5',\
                 'GFDL-CM3', 'CSIRO-Mk3.6', 'IPSL-CM6A-LR', 'CNRM-CM6.1', 'CESM2', 'CanESM5']
    data = pd.DataFrame(ew_mphi.T, columns = mod_names)
    corr = data.corr()
    
    #Inset for model correlation
    #from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # cax = inset_axes(axin,
    #                  width="40%",  # width: 40% of parent_bbox width
    #                  height="5%",  # height: 10% of parent_bbox height
    #                  loc='lower right',
    #                  bbox_to_anchor=(0, 1.1, 1, 1),
    #                  bbox_transform=axin.transAxes,
    #                  borderpad=0,
    #                  )
    
    # axin = sns.heatmap(
    #     corr, alpha=0.8,
    #     vmin=0.8, vmax=1, center=0.9,
    #     cmap=pplt.Colormap('Fire'),
    #     square=True, ax=axin,
    #     cbar_ax=cax, cbar_kws={'orientation': 'horizontal'}
    # )
    # axin.set_xticklabels(
    #     axin.get_xticklabels(),
    #     rotation=90
    # );

    #save figure
    plt.savefig('SM_Fig2.png', dpi=300, transparent=False, bbox_inches="tight")
