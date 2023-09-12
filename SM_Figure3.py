#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__title__ = "Observed trends and phi contour maps for Central Pacific windstress and Southern Ocean SSTa"
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
## Load modules
import os
import sys

import warnings
warnings.filterwarnings('ignore')

import argparse
import numpy as np
import numpy.ma as ma
import xarray as xr
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

import pymannkendall as mk
import proplot as pplt
from adjustText import adjust_text
import klepto
from scipy.stats import iqr
import klepto
from sklearn.linear_model import LinearRegression

pplt.rc['tick.lenratio']=0.02
pplt.rc['savefig.transparent']=True

plt.rcParams.update({'hatch.color': '#363636'})

#My modules 
from plot_maps import *
from load_le_data import * 
from func import *

#============================================================
### Execute script

if __name__ == "__main__"from plot_maps import *
    #Load from klepto
    db = klepto.archives.dir_archive('mphi_ws_CP',serialized=True,cached=False)
    ws_mphi = db['mphi'] 
    mti = b['mti']
    ws_mlb = db['mlb']
    ws_mub = db['mub']

    db = klepto.archives.dir_archive('mphi_ws_CP',serialized=True,cached=False)
    so_mphi = db['mphi'] 
    so_mlb = db['mlb']
    so_mub = db['mub']

    #Model agreement contours 50% - 7/15, 80% - 12/15, 100% - 15/15
    ws_c100 = []
    for l in range(ws_mphi.shape[1]):
        if int(sum(abs(i) > 2 for i in ws_mphi[:,l])) >= int(100*len(tos_moddata)/100):
            ws_c100.append(1)
        else:
            ws_c100.append(0)
            
    so_c50 = []
    for l in range(so_mphi.shape[1]):
        if sum(abs(i) > 2 for i in so_mphi[:,l]) >= int(50*len(tos_moddata)/100):
            so_c50.append(1)
        else:
            so_c50.append(0)
            
    so_c80 = []
    for l in range(so_mphi.shape[1]):
        if sum(abs(i) > 2 for i in so_mphi[:,l]) >= int(80*len(tos_moddata)/100):
            so_c80.append(1)
        else:
            so_c80.append(0)

    # Create dataframe
    shape = int(np.sqrt(len(ws_trends)))
    ws_trendsr = np.reshape(ws_trends,(shape,shape)).T #Reshape to 2D matrix and transpose
    ws_dft = pd.DataFrame(ws_trendsr, columns=[str(v) for v in years])

    ws_c100r = np.reshape(ws_c100,(shape,shape)).T #Reshape to 2D matrix and transpose
    ws_dc100 = pd.DataFrame(ws_c100r, columns=[str(v) for v in years])

    so_trendsr = np.reshape(so_trends,(shape,shape)).T #Reshape to 2D matrix and transpose
    so_dft = pd.DataFrame(so_trendsr, columns=[str(v) for v in years])

    so_c50r = np.reshape(so_c50,(shape,shape)).T #Reshape to 2D matrix and transpose
    so_dc50 = pd.DataFrame(so_c50r, columns=[str(v) for v in years])

    so_c80r = np.reshape(so_c80,(shape,shape)).T #Reshape to 2D matrix and transpose
    so_dc80 = pd.DataFrame(so_c80r, columns=[str(v) for v in years])

    #Change index values to years
    ws_dft = ws_dft.set_index(np.arange(start_year,end_year))
    ws_dc100 = ws_dc100.set_index(np.arange(start_year,end_year))

    so_dft = so_dft.set_index(np.arange(start_year,end_year))
    so_dc50 = so_dc50.set_index(np.arange(start_year,end_year))
    so_dc80 = so_dc80.set_index(np.arange(start_year,end_year))

    # PLOT
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
    mask = np.triu(np.ones_like(ws_trendsr, dtype=bool))
    # Draw the heatmap with the mask and correct aspect ratio
    vmin = -0.02
    vmax = 0.02
    center= 0
    h1 = sns.heatmap(ws_dft, mask=mask, vmin=vmin, vmax=vmax, cmap=pplt.Colormap('NegPos'), center=center,\
            xticklabels=4, yticklabels=4, linecolor='white', linewidths=0.5,
            square=True, ax=ax1, cbar_ax=cax1) #\u03A6 - phi
    cax1.set_ylabel('Windstress trend [N m$^{-2}$ decade$^{-1}$]', fontsize=14,labelpad=-75)
    cax1.tick_params(labelsize=12)
    h1.set_yticklabels(h1.get_yticklabels(), rotation = 0, fontsize = 12)
    h1.set_xticklabels(h1.get_xticklabels(), rotation = 90, fontsize = 12)
    cax1.yaxis.set_ticks_position('left')

    #Set labels
    ax1.set_xlabel('Start year',fontsize=14)
    ax1.set_ylabel('End year',fontsize=14)

    ax1.set_title('(a) Central Pacific surface zonal wind stress',fontsize=16,pad=15) 

    #Set limits 
    h1.set_xlim(0,71-19)
    h1.set_ylim(19,71)
    # h1.xaxis.set_ticks_position('top') 
    # h1.xaxis.set_label_position('top') 
    h1.invert_yaxis()

    # # # # #Smooth contours
    # # import scipy.ndimage as ndimage
    # # pw=1.007
    # # pw=1.0
    # # ws_dc80s = ndimage.zoom(ws_dc80, pw)
    # # ws_dc100s = ndimage.zoom(ws_dc100, pw)
    # # from skimage.transform import resize
    # # ws_dc80s= resize(ws_dc80s,(72,72),order=1) 
    # # ws_dc100s= resize(ws_dc100s,(72,72),order=1) 

    #Overlay phi contours
    levels=[1]
    c1 = h1.contour(np.arange(.5, ws_dc100.shape[1]), np.arange(.5, ws_dc100.shape[0]), ws_dc100, \
                    levels=levels, interpolation='nearest', colors='#5D3A9B',linewidth=1.)

    #Time intervals in lit
    def plot_scatter(y1,y2,text=''):
        ticks = np.arange(0.5,71.5)
        years=np.arange(start_year,end_year)
        x = ticks[np.where(years==int(y1))]
        y = ticks[np.where(years==int(y2))]
    #     ax.scatter(x,y,facecolors='none',edgecolors='k',s=80, marker="s", linewidth=2)
        ax1.text(x,y,text,horizontalalignment='center',verticalalignment='center',fontsize=15,weight='bold')


    plot_scatter(1992,2011,text='A')
    plot_scatter(1980,2012,text='A')

    # # #Add legend 
    ax1.annotate(r"$\bf{" + "A" + "}$" + " England et al. 2014", xy=(0.46,0.7),xycoords='axes fraction',fontsize=14,
                transform_rotates_text=True) #rotation=46,rotation_mode='anchor'

    # #Add contour legend
    from matplotlib.lines import Line2D
    l100 = Line2D([0], [0], label='100%', color='#5D3A9B')

    leg = ax1.legend(handles=[l100],ncol=1,bbox_to_anchor=[0.95,0.95],fontsize=14,
                    title='% of models with < 5% probability \n of containing the observed trend \n in their distribution')

    leg.get_frame().set_linewidth(0.0)
    plt.setp(leg.get_title(),fontsize=14)


    ##SO
    # Draw the heatmap with the mask and correct aspect ratio
    vmin = -0.1
    vmax = 0.1
    center = 0
    h2 = sns.heatmap(so_dft, vmin=vmin, vmax=vmax, cmap=pplt.Colormap('NegPos'), center=center,
                    xticklabels=4,yticklabels=4, linecolor='white', linewidths=0.5,
            square=True, ax=ax2, cbar_ax = cax2) #\u03A6 - phi ,'label': '$^\circ$C/decade'
    cax2.tick_params(labelsize=12)
    cax2.set_ylabel('SST trend [$^\circ$C decade$^{-1}$]', fontsize=14,labelpad=-75)
    #Set limits 
    h2.set_xlim(0,71-19)
    h2.set_ylim(19,71)
    # h2.xaxis.set_ticks_position('top')
    # h2.xaxis.set_label_position('top') 
    h2.invert_yaxis()
    # h2.yaxis.set_ticks_position('right') 
    # h2.yaxis.set_label_position('right') 
    h2.set_yticklabels(h2.get_yticklabels(), rotation = 0, fontsize = 12)
    h2.set_xticklabels(h2.get_xticklabels(), rotation = 90, fontsize = 12)
    cax2.yaxis.set_ticks_position('left')

    #Set labels
    ax2.set_xlabel('Start year',fontsize=14)
    ax2.set_ylabel('End year',fontsize=14)

    ax2.set_title('(b) Southern Ocean SSTs',fontsize=16, pad=15)
    # ax2.set_title('(b) Southern Ocean SSTs',fontsize=16,y=-0.18)

    ax2.set_zorder(-1)

    #Overlay phi contours
    levels=[1]
    c1 = h2.contour(np.arange(.5, so_dc50.shape[1]), np.arange(.5, so_dc50.shape[0]), so_dc50, \
                    levels=levels, interpolation='gaussian', colors='#5D3A9B',linewidth=1.)
    c2 = h2.contour(np.arange(.5, so_dc80.shape[1]), np.arange(.5, so_dc80.shape[0]), so_dc80, \
                    levels=levels, interpolation='nearest', colors='#D55E00',linewidth=1.)

    l50 = Line2D([0], [0], label='50%', color='#5D3A9B')
    l80 = Line2D([0], [0], label='80%', color='#D55E00')
    leg = ax2.legend(handles=[l50, l80],ncol=2,bbox_to_anchor=[0.95,0.95],fontsize=14,
                    title='\n \n \n',
                    frameon=False)

    leg.get_frame().set_linewidth(0.0)
    plt.setp(leg.get_title(),fontsize=14)

    plt.savefig('SM_Fig3.png', dpi=300, facecolor='white', bbox_inches="tight")
