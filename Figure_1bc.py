#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__title__ = "Observed east-west equatorial Pacific SST gradient and phi heat maps"
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

if __name__ == "__main__":

    #Arrange models according to increasing EffCS 
    mods = [gfdl, miroc6, miroc, giss, cesm1, nor, mpi, canesm2, access, gfdlcm3, csiro, ipsl, cnrm, cesm2, canesm5]
    ecs = [2.44, 2.6 , 2.66, 2.71, 2.94, 3.03, 3.63, 3.7 , 3.88, 3.95, 4.09, 4.7 , 4.9 ,5.15, 5.64]
    mod_units = ['K', 'C', 'C', 'C', 'C', 'C', 'K', 'K', 'C', 'K', 'K', 'C', 'C', 'K', 'C']

    #convert to degC
    mod_data = mods
    for i in range(len(mods)):
        if mod_units[i] == 'K':
            mod_data[i] = mods[i] - 273.15
        elif mod_units == 'C':
            pass

    db = klepto.archives.dir_archive('mphi_sst_E-W',serialized=True,cached=False)
    mphi = db['mphi'] 
    mti = b['mti']
    mlb = db['mlb']
    mub = db['mub']

    #Define model agreement 
    pct = [(100/len(mod_data))*n for n in np.arange(1,len(mod_data)+1)]
    # pct=[0,20,40,60,80,100]
    c = [ [] for _ in range(len(pct)) ]
    for n in range(len(c)):
        for l in range(ew_mphi.shape[1]):
            if sum(abs(i) > 2 for i in ew_mphi[:,l]) >= int(pct[n]*len(mod_data)/100):
                c[n].append(pct[n])
            else:
                c[n].append(0.)
    c = np.stack(c)

    ca = np.empty(len(c[0]))
    for i in range(len(ca)):
        ca[i] = np.nanmax(c[:,i])

    levels = np.unique(ca)[2:]
    #Number of models 
    levels_label = [int((v/100)*len(mod_data)) for v in levels]

    shape = int(np.sqrt(len(ew_trends)))
    cr = np.reshape(ca,(shape,shape)).T #Reshape to 2D matrix and transpose
    df_cr = pd.DataFrame(cr, columns=[str(v) for v in years])
    df_cr.replace(0, np.nan, inplace=True)

    # Create dataframe
    ew_trendsr = np.reshape(ew_trends,(shape,shape)).T #Reshape to 2D matrix and transpose
    df_ew = pd.DataFrame(ew_trendsr, columns=[str(v) for v in years])
    df_ew.replace(0, np.nan, inplace=True)

    #Change index values to years
    df_ew = df_ew.set_index(np.arange(start_year,end_year))
    df_cr = df_cr.set_index(np.arange(start_year,end_year))

    ## PLOT 

    fig = plt.figure(figsize=(19,8))
    widths = [0.03,1,1,0.03]
    heights = [1]
    spec = gridspec.GridSpec(ncols=4, nrows=1, width_ratios=widths,
                            height_ratios=heights,hspace=0.,wspace=0.3)
    ax1 = fig.add_subplot(spec[0, 1])
    ax2 = fig.add_subplot(spec[0, 2])
    cax1 = fig.add_subplot(spec[0, 0])
    cax2 = fig.add_subplot(spec[0, 3])

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(ew_trendsr, dtype=bool))
    # Draw the heatmap with the mask and correct aspect ratio
    vmin = -0.4
    vmax = 0.4
    center= 0
    h1 = sns.heatmap(df_ew, vmin=vmin, vmax=vmax, cmap=pplt.Colormap('NegPos'), center=center,\
            xticklabels=4, yticklabels=4,
            square=True, linewidths=.5, ax=ax1, cbar_ax=cax1) #\u03A6 - phi
    #format axis 
    h1.set_yticklabels(h1.get_yticklabels(), rotation = 0, fontsize = 12)
    h1.set_xticklabels(h1.get_xticklabels(), rotation = 90, fontsize = 12)
    #Set labels
    ax1.set_xlabel('Start year',fontsize=14)
    ax1.set_ylabel('End year',fontsize=14)

    #Set limits 
    h1.set_xlim(0,71-19)
    # h1.set_xlim(19,71)
    h1.set_ylim(19,71)
    # h1.set_ylim(0,71-19)
    # h1.invert_xaxis()
    h1.invert_yaxis()
    # h1.xaxis.set_ticks_position('top') 
    # h1.xaxis.set_label_position('top') 

    #format colorbar
    cax1.set_ylabel('SST trend [$^\circ$C decade$^{-1}$]', fontsize=14,labelpad=-60)
    cax1.tick_params(labelsize=12)
    cax1.yaxis.set_ticks_position('left')

    #set title
    ax1.set_title('(b) East-West Pacific gradient in observations',fontsize=16,y=-0.18)


    # Model disagreement
    # Draw the heatmap with the mask and correct aspect ratio
    vmin = levels[0]
    vmax = levels[-1]
    center = 60 #50
    cmap = pplt.Colormap("BuPu",
                        samples=len(levels)-1, discrete=True)

    h2 = sns.heatmap(ew_dc, alpha=0.6,
                    vmin=vmin, vmax=vmax, cmap=cmap,
                    xticklabels=4,yticklabels=4, square=True, linewidths=.5,
                    ax=ax2, cbar_ax = cax2)

    #Format axis
    #Set limits 
    h2.set_xlim(0,71-19)
    h2.set_ylim(19,71)
    # h2.set_ylim(0,71-19)
    # h2.set_xlim(19,71)
    # h2.xaxis.set_ticks_position('top')
    h2.yaxis.set_ticks_position('right')
    # h2.xaxis.set_label_position('top') 
    h2.yaxis.set_label_position('right') 
    h2.set_yticklabels(h2.get_yticklabels(), rotation = 0, fontsize = 12)
    h2.set_xticklabels(h2.get_xticklabels(), rotation = 90, fontsize = 12)

    ax2.invert_xaxis()
    ax2.invert_yaxis()

    #Set labels
    ax2.set_xlabel('Start year',fontsize=14)
    ax2.set_ylabel('End year',fontsize=14)

    #format colorbar
    #cax2.set_ylabel('[%]', fontsize=14)
    cax2.set_ylabel('Number of models', fontsize=14)
    cax2.tick_params(labelsize=12)
    labels = levels_label[1:]
    cax2.set_yticklabels([int(v) for v in labels])

    # Manually specify colorbar labelling after it's been generated
    colorbar = h2.collections[0].colorbar
    colorbar.set_ticks(levels[1:] - np.diff(levels)[0]/2)

    #Time intervals in lit
    def plot_scatter(y1,y2,text=''):
        ticks = np.arange(0.5,71.5)
        years=np.arange(start_year,end_year)
        x = ticks[np.where(years==int(y1))]
        y = ticks[np.where(years==int(y2))]
    #     ax.scatter(x,y,facecolors='none',edgecolors='k',s=50, marker="s", linewidth=2)
        ax2.text(x,y,text,horizontalalignment='center',verticalalignment='center',fontsize=15,weight='bold')


    #Olonscheck 2020
    plot_scatter(1975,2004,text='A')
    plot_scatter(1995,2014,text='A')

    #Seager et al. 2019
    plot_scatter(1958,2008,text='C')
    plot_scatter(1958,2017,text='C')

    #Watanabe 2021
    plot_scatter(1951,1990,text='B')
    plot_scatter(1951,2010,text='B')

    #Wills 2022
    plot_scatter(1979,2020,text='D')

    import matplotlib.patches as patches
    h2.add_patch(
        patches.Rectangle(
            (1, 41),
            1.0,
            19.0,
            edgecolor='k',
            fill=False,
            lw=1
        ) )

    h2.add_patch(
        patches.Rectangle(
            (8, 59),
            1.0,
            8.0,
            edgecolor='k',
            fill=False,
            lw=1
        ) )

    #add circles
    import matplotlib.patches as patches
    h1.add_patch(
        patches.Circle(
            (40.5,65.5),radius=1,
            edgecolor='k',
            fill=False,
            lw=1
        ) )

    h1.add_patch(
        patches.Circle(
            (20.5,45.5),radius=1,
            edgecolor='k',
            fill=False,
            lw=1
        ) )

    h2.add_patch(
        patches.Circle(
            (40.5,65.5),radius=1,
            edgecolor='k',
            fill=False,
            lw=1
        ) )

    h2.add_patch(
        patches.Circle(
            (20.5,45.5),radius=1,
            edgecolor='k',
            fill=False,
            lw=1
        ) )

    #Add legend 
    ax2.annotate(r"$\bf{" + "A" + "}$" + " Olonscheck et al. 2020", xy=(-0.06,0.1),  #(0.21,0.71)
                xycoords='axes fraction',fontsize=14, rotation=45,
                rotation_mode='anchor', transform_rotates_text=True)
    ax2.annotate(r"$\bf{" + "B" + "}$" + " Watanabe et al. 2021", xy= (-0.025, 0.06), #(0.51,0.41)
                xycoords='axes fraction',fontsize=14, rotation=45,
                rotation_mode='anchor', transform_rotates_text=True)
    ax2.annotate(r"$\bf{" + "C" + "}$" + " Seager et al. 2019", xy=(0.4,0.55), #(0.18,0.68)
                xycoords='axes fraction',fontsize=14, rotation=45,
                rotation_mode='anchor', transform_rotates_text=True)
    ax2.annotate(r"$\bf{" + "D" + "}$" + " Wills et al. 2022", xy=(0.435, 0.514), #(0.47,0.19),
                xycoords='axes fraction',fontsize=14, rotation=45,
                rotation_mode='anchor', transform_rotates_text=True)

    ax2.set_title('(c) CMIP model discrepancy with observations',fontsize=16,y=-0.18)


    # #set titles 
    # ax1.annotate('(b) East-West Pacific gradient in observations', xy = (0.34,0.88),
    #               xycoords='axes fraction',fontsize=16,rotation=-45,rotation_mode='anchor',transform_rotates_text=True)

    # ax2.annotate('(c) CMIP model discrepancy with observations', xy = (0.1,0.35),
    #               xycoords='axes fraction',fontsize=16,rotation=45,rotation_mode='anchor',transform_rotates_text=True)

    ax2.step(np.arange(0,54),np.arange(18,72),color='#CBC3E3')

    ax2.step(np.arange(start_year,end_year),np.arange(start_year,end_year))

    plt.savefig('Fig_1bc.png',
                dpi=300, facecolor='white', bbox_inches="tight")