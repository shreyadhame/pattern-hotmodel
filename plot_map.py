#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__title__ = ""
__reference__ = ""
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreya.dhame@mpimet.mpg.de"

#==============================================================================
#General modules
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.geoaxes import GeoAxes
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib import gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import numpy as np
from scipy.interpolate import interp2d

#==============================================================================
class MidpointNormalize(mcolors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        mcolors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


def plot_maps_gridspec(ax,
var,lon,lat,levels,cmap,mp=0.,central_longitude=180.,\
extent=False,lat1=-90.,lat2=90.,lon1=0.,lon2=360.,lat_step=10,lon_step=60,\
ticks=True,\
land=True,
title=['Give Subplot Titles Here'],fontsize=12,pad=2,loc_title='left'
):

    transform=ccrs.PlateCarree()
    ax.coastlines(lw=1.)
    if extent==True:
        ax.set_extent([lon1,lon2,lat1,lat2],crs=transform)
    elif extent==False:
        pass

    if ticks==True:
        #yticks = np.arange(lat1, lat2+1, lat_step)
        yticks = np.arange(lat1,lat2+1,lat_step)
        xticks = np.arange(lon1, lon2+1, lon_step)
        ax.set_xticks(xticks, crs=transform)
        ax.set_yticks(yticks, crs=transform)

        # ax.xaxis.set_tick_params(labelsize=fontsize)
        # ax.yaxis.set_tick_params(labelsize=fontsize)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
    else:
        pass
    
    vmin=levels[0]
    vmax=levels[-1]
    norm=MidpointNormalize(midpoint=mp,vmin=vmin,vmax=vmax)
    b = ax.contourf(lon, lat, var,levels=levels,
                transform=transform,cmap=cmap, extend='both',
                norm=norm
                )

    if land==True:
        ax.add_feature(cfeature.LAND, facecolor='#B1B1B1')
    elif land==False:
        pass

    ax.set_title(title,pad=pad,loc=loc_title)
