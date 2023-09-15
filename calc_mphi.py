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

from sklearn.linear_model import LinearRegression

pplt.rc['tick.lenratio']=0.02
pplt.rc['savefig.transparent']=True

plt.rcParams.update({'hatch.color': '#363636'})

#My modules 
from load_le_data import * 
from func import *

#============================================================
### Execute script

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('var') #sst | windstress
    parser.add_argument('reg') #EP | WP | E-W | SO | CP
    parser.add_argument('type_sigma') #std | iqr

    args = parser.parse_args()

    var = str(args.var)
    reg = str(args.reg)
    type_sigma = str(args.type_sigma)

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
    
    if reg == 'SO':
        lat1 = 5
        lat2 = -5
        lon1 = -180%360
        lon2 = -80%360  
    elif reg == 'E-W':
        #Select equatorial eastern Pacific region
        lat1 = 5
        lat2 = -5
        lone1 = -180%360
        lone2 = -80%360
        
        obs_reg1, lons, lats = sel_reg(obs,lon,lat,lat1=lat2,lat2=lat1,lon1=lone1,lon2=lone2)
        mod_reg1 = [sel_reg(v,lon,lat,lat1=lat2,lat2=lat1,lon1=lone1,lon2=lone2)[0] for v in mod_data]
        #Take spatial mean
        obs_ts1 = wgt_mean(obs_reg1,lons,lats)
        mod_ts1=[]
        for i in range(len(mod_reg1)):
            mod_ts1.append([wgt_mean(ma.masked_where(v<-5,v),lons,lats) for v in mod_reg1[i]])

        #Select equatorial western Pacific region
        lonw1 = 110
        lonw2 = 180

        obs_reg2, lons, lats = sel_reg(obs,lon,lat,lat1=lat2,lat2=lat1,lon1=lonw1,lon2=lonw2)
        mod_reg2 = [sel_reg(v,lon,lat,lat1=lat2,lat2=lat1,lon1=lonw1,lon2=lonw2)[0] for v in mod_data]
        #Take spatial mean
        obs_ts2 = wgt_mean(obs_reg2,lons,lats) 
        mod_ts2=[]
        for i in range(len(mod_reg2)):
            mod_ts2.append([wgt_mean(ma.masked_where(v<0,v),lons,lats) for v in mod_reg2[i]])

        #East-west gradient
        reg_obs = obs_ts1 - obs_ts2
        reg_mods = []
        for i in range(len(mod_ts2)):
            reg_mods.append(np.stack([np.array(mod_ts1[i][j] - mod_ts2[i][j]) for j in range(len(mod_ts2[i]))])) 

    #Calculate observed trends for all time periods (E-W gradient)
    start_year = 1950 
    end_year = 2021

    trends = []
    for i in range(start_year,end_year):
        for j in range(start_year,end_year):
            ind1 = np.where(time.dt.year==i)[0][0] #Index of start year
            ind2 = np.where(time.dt.year==j)[0][0] #Index of end year
            #Select a chunk
            chunk = reg_obs[ind1:ind2] 
            #Calculate trends only for >19 year chunks
            if len(chunk) >= 19: #*12: 
                trends.append(mk_test(chunk)[-1]*10) #decadal trend
            else:
                trends.append(np.nan)

    #Calculate modeled trends for all time periods periods (E-W gradient)
    #Ensemble mean of each model
    modsm = [v.mean(axis=0) for v in reg_mods]

    mtrends = [[] for _ in range(len(modsm))]
    for k in range(len(modsm)):
        for i in range(start_year,end_year):
            for j in range(start_year,end_year):
                ind1 = np.where(time.dt.year==i)[0][0] #Index of start year
                ind2 = np.where(time.dt.year==j)[0][0] #Index of end year
                #Select a chunk
                chunk = modsm[k][ind1:ind2] 
                #Calculate trends only for >19 year chunks
                if len(chunk) >= 19: #*12: 
                    mtrends[k].append(mk_test(chunk)[-1]*10) #decadal trend
                else:
                    mtrends[k].append(np.nan)

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
    mphi = np.stack(mphi)
    mlb = np.stack(mlb)
    mub = np.stack(mub)
    mti = np.stack(mti)

#Save to klepto
db = klepto.archives.dir_archive('mphi_'+str(var)+'_'+str(reg),serialized=True,cached=False)
db['mphi'] = mphi
db['mti'] = mti
db['mlb'] = mlb
db['mub'] = mub