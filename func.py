#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__title__ = "Functions"
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
def lag1_acf(x, nlags=1):
    """
    Lag 1 autocorrelation
    Parameters
    ----------
    x : 1D numpy.ndarray
    nlags : Number of lag
    Returns
    -------
    acf : Lag-1 autocorrelation coefficient
    """
    y = x - np.nanmean(x)
    n = len(x)
    d = n * np.ones(2 * n - 1)

    acov = (np.correlate(y, y, 'full') / d)[n - 1:]
    acf = acov[:nlags]/acov[0]
    return acf

def mk_test(x, a=0.10):
    """
    Mann-Kendall test for trend
    Parameters
    ----------
    x : 1D numpy.ndarray
    a : p-value threshold
    Returns
    -------
    trend : tells the trend (increasing, decreasing or no trend)
    h : True (if trend is present or Z-score statistic is greater than p-value) or False (if trend is absent)
    p : p-value of the significance test
    z : normalized test statistics
    Tau : Kendall Tau (s/D)
    s : Mann-Kendal's score
    var_s : Variance of s
    slope : Sen's slope
    """
    #Calculate lag1 acf
    acf = lag1_acf(x)

    r1 = (-1 + 1.96*np.sqrt(len(x)-2))/len(x)-1
    r2 = (-1 - 1.96*np.sqrt(len(x)-2))/len(x)-1
    if (acf > 0) and (acf > r1):
        #remove serial correlation
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.yue_wang_modification_test(x)
    elif (acf < 0) and (acf < r2):
        #remove serial correlation
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.yue_wang_modification_test(x)
    else:
        #Apply original MK test
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(x)
    return h, p, z, Tau, s, var_s, slope


def calc_phi(mod,obs,type_sigma='std'):
    """
    Calculates phi
    Parameters
    ----------
    mod, obs : 1-D arrays
    type_sigma : 'std' or 'iqr'
    Returns
    -------
    phi : modelled-observed discrepancy 
    lb, ub : lower and upper bounds of the interquartile range

    """
    #Calculate ensemble mean
    modm = np.nanmean(mod,axis=0)
    #Calculate trends
    trnd_modm = mk_test(modm)[-1]*10 
    trnd_obs = mk_test(obs)[-1]*10 
    trnd_mod = np.stack([mk_test(v)[-1]*10 for v in mod]) 
    #Calculate sigma
    if type_sigma == 'std':
        sigma = np.nanstd(trnd_mod)
        lb = []
        ub = []
    elif type_sigma == 'iqr':
        sigma = iqr(trnd_mod, rng=(5, 95))
        lb,ub = np.quantile(trnd_mod/sigma,[0.05,0.95])
    #Calculate phi
    phi = (trnd_modm - trnd_obs)/sigma
    return phi,lb,ub

def sel_reg(var,lon,lat,lat1,lat2,lon1,lon2):
    #Revised lon, lat, region
    lats = lat.sel(lat=slice(lat1,lat2))
    lons = lon.sel(lon=slice(lon1,lon2))
    reg = var.sel(lat=slice(lat1,lat2),lon=slice(lon1,lon2))
    return reg,lons,lats

def wgt_mean(var,lon,lat):
    """
    Calculate weighted mean
    Parameters
    ----------
    var : 3-D array 
    lat, lon : 1-D arrays
    """
    #Mask nans
    var_ma = ma.masked_invalid(var)
    #Weight matrix
    #lat60 = lat.sel(lat=slice(60,-60))
    wgtmat = np.cos(np.tile(abs(lat.values[:,None])*np.pi/180,(1,len(lon))))[np.newaxis,...] #(time,lat,lon)
    #Apply
    #var_mean = np.ma.sum((var_ma*wgtmat*~var_ma.mask)/(np.ma.sum(wgtmat * ~var_ma.mask)))
    var_mean = var_ma*wgtmat*~var_ma.mask
    var_m = np.nanmean(np.nanmean(var_mean,axis=-1),axis=-1)
    return var_m