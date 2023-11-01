#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__title__ = "Calculate phi for all model ensembles for SST, windstress"
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
## Load modules
import os
import sys

import warnings
warnings.filterwarnings('ignore')

from adjustText import adjust_text
import argparse
import klepto
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import proplot as pplt
import pymannkendall as mk
from scipy import stats
from scipy.stats import iqr
import seaborn as sns
from sklearn.linear_model import LinearRegression
import xarray as xr

#My modules 
from load_le_data import * 
from func import *

#============================================================
### Execute script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('type_sigma') #std | iqr (standard deviation or interquartile range)

    args = parser.parse_args()

    type_sigma = str(args.type_sigma)

    #============================================================
    ##East-west Pacific
    #Eastern Pacific
    lat1 = 5
    lat2 = -5
    lone1 = -180%360
    lone2 = -80%360
    #Western Pacific
    lonw1 = 110
    lonw2 = 180

    #Select region
    obs_reg1, lons, lats = selreg(obs,lon,lat,lat1=lat2,lat2=lat1,lon1=lone1,lon2=lone2)
    mod_reg1 = [selreg(v,lon,lat,lat1=lat2,lat2=lat1,lon1=lone1,lon2=lone2)[0] for v in mod_data]
    
    #Take spatial mean
    obs_ts1 = wgt_mean(obs_reg1,lons,lats)
    mod_ts1=[]
    for i in range(len(mod_reg1)):
        mod_ts1.append([wgt_mean(ma.masked_where(v<-5,v),lons,lats) for v in mod_reg1[i]])

    #Select region
    obs_reg2, lons, lats = selreg(obs,lon,lat,lat1=lat2,lat2=lat1,lon1=lonw1,lon2=lonw2)
    mod_reg2 = [selreg(v,lon,lat,lat1=lat2,lat2=lat1,lon1=lonw1,lon2=lonw2)[0] for v in mod_data]
    mod_reg2[13] = ma.masked_where(mod_reg2[13] < 6., mod_reg2[13])
    
    #Take spatial mean
    obs_ts2 = wgt_mean(obs_reg2,lons,lats) 
    mod_ts2=[]
    for i in range(len(mod_reg2)):
        mod_ts2.append([wgt_mean(ma.masked_where(v<0,v),lons,lats) for v in mod_reg2[i]])

    #Calculate east-west gradient
    ew_obs = obs_ts1 - obs_ts2
    ew_mods = []
    for i in range(len(mod_ts2)):
        ew_mods.append(np.stack([np.array(mod_ts1[i][j] - mod_ts2[i][j]) for j in range(len(mod_ts2[i]))])) 


    #Calculate observed trends for chunks of periods (E-W)
    start_year = 1950
    end_year = 2021
    ew_trends = []
    for i in range(start_year,end_year):
        for j in range(start_year,end_year):
            ind1 = np.where(time.dt.year==i)[0][0] #Index of start year
            ind2 = np.where(time.dt.year==j)[0][0] #Index of end year
            chunk = ew_obs[ind1:ind2] #Select a chunk
            if len(chunk) >= 19: #*12: #Calculate trends only for >20 year chunks
                ew_trends.append(mk_test(chunk)[-1]*10) #len(chunk)) or 10 for decadal trend
            else:
                ew_trends.append(np.nan)


    #Calculate modeled trends for chunks of periods (E-W)
    #Model mean 
    ew_modsm = [v.mean(axis=0) for v in ew_mods]
    
    ew_mtrends = [[] for _ in range(len(ew_modsm))]
    for k in range(len(ew_modsm)):
        for i in range(start_year,end_year):
            for j in range(start_year,end_year):
                ind1 = np.where(time.dt.year==i)[0][0] #Index of start year
                ind2 = np.where(time.dt.year==j)[0][0] #Index of end year
                chunk = ew_modsm[k][ind1:ind2] #Select a chunk
                if len(chunk) >= 19: #*12: #Calculate trends only for >20 year chunks
                    ew_mtrends[k].append(mk_test(chunk)[-1]*10) #len(chunk)) or 10 for decadal trend
                else:
                    ew_mtrends[k].append(np.nan)


    #Calculate phi for all time periods
    #Create empty arrays
    mphi = [] #Phi for all models
    mti = [] #time periods
    mlb = [] #lower bound of the ensemble ditribution
    mub = [] #upper bound of the ensemble distribution
    #Loop through all models
    for m in range(len(reg_mods)):
        phi=[]
        ti = []
        lb = []
        ub = []
        start_year = 1950
        end_year = 2021
        for i in range(start_year,end_year):
            for j in range(start_year,end_year):
                ind1 = np.where(time.dt.year==i)[0][0] #Index of start year
                ind2 = np.where(time.dt.year==j)[0][0] #Index of end year
                chunk_obs = reg_obs[ind1:ind2]
                chunk_mod = reg_mods[m][:,ind1:ind2]
                if len(chunk_obs) >= 19:
                    #Calculate phi
                    phi.append(calc_phi(chunk_mod,chunk_obs,type_sigma = type_sigma)[0])
                    lb.append(calc_phi(chunk_mod,chunk_obs,type_sigma = type_sigma)[1])
                    ub.append(calc_phi(chunk_mod,chunk_obs,type_sigma = type_sigma)[2])
                    ti.append(str(i)+'-'+str(j))
                else:
                    phi.append(0.)
                    lb.append(0.)
                    ub.append(0.)
                    ti.append('')
        mphi.append(np.array(phi))
        mlb.append(np.array(lb))
        mub.append(np.array(ub))
        mti.append(np.array(ti))
    #Stack arrays of all models
    ew_mphi = np.stack(mphi)
    mlb = np.stack(mlb)
    mub = np.stack(mub)
    mti = np.stack(mti)

    db = klepto.archives.dir_archive('mphi',serialized=True,cached=False)
    db['ew_trends'] = ew_trends
    db['ew_mtrends'] = ew_mtrends
    db['ew_mphi'] = ew_mphi
    db['mti'] = mti
    db['mlb'] = mlb
    db['mub'] = mub

    #============================================================
    #Calculate r2 and slope for east-west gradient and EffCS
    ew_r = []
    ew_slope = []
    ew_intercept = []
    for l in range(ew_mphi.shape[1]): 
        if np.count_nonzero(ew_mphi[:,l])==0:
            ew_r.append(np.nan)
            ew_slope.append(np.nan)
            ew_intercept.append(np.nan)
        else:
            #Calculate R2
            r_value = stats.linregress(ew_mphi[:,l],ecs)[2]
            ew_r.append(r_value**2)
            #Calculate slope
            ew_slope.append(stats.linregress(ew_mphi[:,l],ecs)[0])
            #Calculate intercept
            ew_intercept.append(stats.linregress(ew_mphi[:,l],ecs)[1])

    #Save to klepto
    db['ew_r'] = ew_r  
    db['ew_slope'] = ew_slope 
    db['ew_intercept'] = ew_intercept 

    #============================================================
    #Southern Ocean SSTs
    lat1 = -45
    lat2 = -65
    lon1 = 0
    lon2 = 360

    so_obs_reg, lons, lats = sel_reg(tos_obs,lon,lat,lat1=lat2,lat2=lat1,lon1=lon1,lon2=lon2)
    so_mod_reg = [sel_reg(v,lon,lat,lat1=lat2,lat2=lat1,lon1=lon1,lon2=lon2)[0] for v in tos_moddata]
    #Take spatial mean
    so_obs_ts = wgt_mean(so_obs_reg,lons,lats)
    so_mod_ts=[]
    for i in range(len(so_mod_reg)):
        so_mod_ts.append(np.stack([wgt_mean(ma.masked_where(v<-5,v),lons,lats) for v in so_mod_reg[i]]))
    so_obs = so_obs_ts
    so_mods = so_mod_ts

    #Define time
    years=np.unique(time.dt.year)

    #Calculate observed trends for chunks of periods (E-W)
    start_year = 1950
    end_year = 2021

    so_trends = []
    for i in range(start_year,end_year):
        for j in range(start_year,end_year):
            ind1 = np.where(time.dt.year==i)[0][0] #Index of start year
            ind2 = np.where(time.dt.year==j)[0][0] #Index of end year
            chunk = so_obs[ind1:ind2] #Select a chunk
            if len(chunk) >= 19: #*12: #Calculate trends only for >20 year chunks
                so_trends.append(mk_test(chunk)[-1]*10) #len(chunk)) or 10 for decadal trend
            else:
                so_trends.append(np.nan) 

    
    #Calculate phi for chunks of periods
    #Create empty array
    so_mphi = []
    #Select chunks
    for m in range(len(so_mods)):
        so_phi=[]
        start_year = 1950
        end_year = 2021
        for i in range(start_year,end_year):
            for j in range(start_year,end_year):
                ind1 = np.where(time.dt.year==i)[0][0] #Index of start year
                ind2 = np.where(time.dt.year==j)[0][0] #Index of end year
                chunk_obs = so_obs[ind1:ind2]
                chunk_mod = so_mods[m][:,ind1:ind2]
                if len(chunk_obs) >= 19: #Calculate trends only for >20 year chunks
                    #Calculate phi
                    so_phi.append(calc_phi(chunk_mod,chunk_obs,type_sigma = type_sigma)[0])
                else:
                    so_phi.append(0.)
        so_mphi.append(np.array(so_phi))
    so_mphi = np.stack(so_mphi)

    #Save to klepto
    db['so_trends'] = so_trends
    db['so_mphi'] = so_mphi

    #============================================================
    #Central Pacific wind stress
    ws_moddata = ws_mods
    
    #Central Pacific 
    lat1 = 5
    lat2 = -5
    lon1 = 180
    lon2 = -150%360

    #Select region
    ws_obs_reg, lons, lats = sel_reg(ws_obs,lon,lat,lat1=lat2,lat2=lat1,lon1=lon1,lon2=lon2)
    ws_obs_reg = np.apply_along_axis(stress,0,ws_obs_reg)
    ws_mod_reg = [sel_reg(v,lon,lat,lat1=lat2,lat2=lat1,lon1=lon1,lon2=lon2)[0] for v in ws_moddata]
    ws_mod_reg[0] = np.apply_along_axis(stress,0,ws_mod_reg[0])
    ws_mod_reg[6] = np.apply_along_axis(stress,0,ws_mod_reg[6])
    #Take spatial mean
    ws_obs_ts = wgt_mean(ws_obs_reg,lons,lats)
    ws_mod_ts=[]
    for i in range(len(ws_mod_reg)):
        ws_mod_ts.append(np.stack([wgt_mean(v,lons,lats) for v in ws_mod_reg[i]]))
    ws_obs = ws_obs_ts
    ws_mods = ws_mod_ts

    #Calculate observed trends for chunks of periods
    start_year = 1950
    end_year = 2021
    
    ws_trends = []
    for i in range(start_year,end_year):
        for j in range(start_year,end_year):
            ind1 = np.where(time.dt.year==i)[0][0] #Index of start year
            ind2 = np.where(time.dt.year==j)[0][0] #Index of end year
            chunk = ws_obs[ind1:ind2] #Select a chunk
            if len(chunk) >= 19: #*12: #Calculate trends only for >20 year chunks
                ws_trends.append(mk_test(chunk)[-1]*10) #len(chunk)) or 10 for decadal trend
            else:
                ws_trends.append(np.nan) 
    
    #Calculate phi for chunks of periods
    #Create empty array
    ws_mphi = []
    mti = []
    #Select chunks
    for m in range(len(ws_mods)):
        ws_phi=[]
        ti = []
        start_year = 1950
        end_year = 2021
        for i in range(start_year,end_year):
            for j in range(start_year,end_year):
                ind1 = np.where(time.dt.year==i)[0][0] #Index of start year
                ind2 = np.where(time.dt.year==j)[0][0] #Index of end year
                chunk_obs = ws_obs[ind1:ind2]
                chunk_mod = ws_mods[m][:,ind1:ind2]
                if len(chunk_obs) >= 19: #Calculate trends only for >20 year chunks
                    #Calculate phi
                    ws_phi.append(calc_phi(chunk_mod,chunk_obs,type_sigma=type_sigma)[0])
                    ti.append(str(i)+'-'+str(j))
                else:
                    ws_phi.append(0.)
                    ti.append('')
        ws_mphi.append(np.array(ws_phi))
        mti.append(np.array(ti))
    ws_mphi = np.stack(ws_mphi)
    mti = np.stack(mti)

    #Save to klepto
    db['ws_trends'] = ws_trends
    db['ws_mphi'] = ws_mphi
