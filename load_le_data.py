#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__title__ = "Load preprocessed large ensemble climate model data"
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

#============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('start_yr') #1950
    parser.add_argument('end_yr') #2020
    args = parser.parse_args()

    start_yr = str(args.start_yr)
    end_yr = str(args.end_yr)

    #Load COBE/ERSSTv5/HadISST1 SST data
    path = '/work/mh0033/m300952/OBS/'
    cobe = xr.open_dataset(path+'COBE/cobe_sst_yr_1950-2021_g025.nc').sst
    ersst = xr.open_dataset(path+'ERSST/ersst_sst_yr_1950-2021_g025.nc').sst
    hadsst = xr.open_dataset(path+'HADISST/HadISST_sst_yr_1950-2021_g025.nc')
    lon=hadsst.lon
    lat=hadsst.lat
    hadsst = hadsst.sst
    obs = xr.concat([cobe,ersst,hadsst],dim="y",join='override').mean(axis=0)

    obs = obs.sel(time=slice(start_yr,end_yr)).squeeze()
    time = hadsst.time[:-1]

    #Load large ensembles (SST)
    path = '/work/mh0033/m300952/CMIP'

    access = xr.open_mfdataset(path+'/ACCESS-ESM1-5/tos/*g025.nc',combine='nested',concat_dim='depth').tos  #9 members degC
    access = access.sel(time=slice(start_yr,end_yr))

    canesm2 = xr.open_mfdataset(path+'/CanESM2/tos/*g025.nc',combine='nested',concat_dim='depth').tos #50 members K
    canesm2 = canesm2.sel(time=slice(start_yr,end_yr))

    canesm5 = xr.open_mfdataset(path+'/CanESM5/tos/*g025.nc',combine='nested',concat_dim='depth').tos #50 members degC
    canesm5 = canesm5.sel(time=slice(start_yr,end_yr))

    cesm1 = xr.open_mfdataset(path+'/CESM1/tos/*g025.nc',combine='nested',concat_dim='depth').SST #40 members degC
    cesm1 = cesm1.sel(time=slice(start_yr,end_yr)).squeeze()

    cesm2 = xr.open_mfdataset(path+'/CESM2/tos/*g025.nc',combine='nested',concat_dim='depth',join='override').tos #106 members 
    # cesm2 = cesm2.sel(time=slice(start_yr,end_yr)).squeeze()
    ind1 = np.where(cesm2.time.dt.year==int(start_yr))[0][0]
    ind2 = np.where(cesm2.time.dt.year==int(end_yr))[0][0] + 1
    cesm2 = cesm2[:,ind1:ind2]

    cnrm = xr.open_mfdataset(path+'/CNRM-CM6-1/tos/*g025.nc',combine='nested',concat_dim='depth').tos #10 members degC
    cnrm = cnrm.sel(time=slice(start_yr,end_yr)).squeeze()

    csiro = xr.open_mfdataset(path+'/CSIRO-Mk3-6/tos/*g025.nc',combine='nested',concat_dim='depth').tos #30 members K
    csiro = csiro.sel(time=slice(start_yr,end_yr))

    gfdlcm3 = xr.open_mfdataset(path+'/GFDL-CM3/ts/*g025.nc',combine='nested',concat_dim='depth').ts #20 members K
    gfdlcm3 = gfdlcm3.sel(time=slice(start_yr,end_yr)).squeeze()

    gfdl = xr.open_mfdataset(path+'/GFDL-ESM2M/tos/*g025.nc',combine='nested',concat_dim='depth').tos #30 members K
    gfdl = gfdl.sel(time=slice(start_yr,end_yr)).squeeze()

    giss = xr.open_mfdataset(path+'/GISS-E2-1-G/tos/*g025.nc',combine='nested',concat_dim='depth').tos #18 members degC
    giss = giss.sel(time=slice(start_yr,end_yr)).squeeze()

    ipsl = xr.open_mfdataset(path+'/IPSL-CM6A-LR/tos/*g025.nc',combine='nested',concat_dim='depth').tos #11 members degC
    ipsl = ipsl.sel(time=slice(start_yr,end_yr)).squeeze()
    
    miroc6 = xr.open_mfdataset(path+'/MIROC6/tos/*g025.nc',combine='nested',concat_dim='depth').tos #50 members degC
    miroc6 = miroc6.sel(time=slice(start_yr,end_yr)).squeeze()

    miroc = xr.open_mfdataset(path+'/MIROC-ES2L/tos/*g025.nc',combine='nested',concat_dim='depth').tos #30 members degC
    miroc = miroc.sel(time=slice(start_yr,end_yr)).squeeze()

    mpi = xr.open_mfdataset(path+'/MPI-ESM/tos/*g025.nc',combine='nested',concat_dim='depth').sst #100 members K
    mpi = mpi.sel(time=slice(start_yr,end_yr)).squeeze()
    mpi = mpi.transpose("depth","time","lat","lon")

    nor = xr.open_mfdataset(path+'/NorCPM1/tos/*g025.nc',combine='nested',concat_dim='depth').tos #30 members degC
    nor = nor.sel(time=slice(start_yr,end_yr)).squeeze()

    #Arrange models according to increasing ECS 
    mods = [gfdl, miroc6, miroc, giss, mpi, nor, canesm2, access, gfdlcm3, csiro, cesm1, ipsl, cnrm, cesm2, canesm5]
    ecs = [2.44, 2.6 , 2.66, 2.71, 2.8, 3.03, 3.7 , 3.88, 3.95, 4.09, 4.1, 4.7 , 4.9 ,5.15, 5.64]
    mod_units = ['K', 'C', 'C', 'C', 'K', 'C', 'K', 'K', 'C', 'K', 'K', 'C', 'C', 'C', 'K', 'C']
    
    #Convert to array
    mods = np.array([np.nan_to_num(v) for v in mods])
    mods_ma = np.copy(mods)

    #convert to degC
    mod_data = np.copy(mods_ma)
    for i in range(len(mods_ma)):
        if mod_units[i] == 'K':
            mod_data[i] = mods_ma[i] - 273.15
        elif mod_units == 'C':
            pass

    #Correct temperatures 
    mod_data[3][3][-6:] = mod_data[3][3][-6:] - 273.15 
    mod_data[3][6][-6:] = mod_data[3][6][-6:] - 273.15
    mod_data[3][9][-6:] = mod_data[3][9][-6:] - 273.15
    mod_data[3][12][-6:] = mod_data[3][12][-6:] - 273.15
    
    #Remove weird values    
    mod_data[0][np.where(mod_data[0] < -2.)] = 0.
    mod_data[3][np.where(mod_data[3] < -2.)] = 0.
    mod_data[6][np.where(mod_data[6] < -2.)] = 0.
    mod_data[7][np.where(mod_data[7] < -2.)] = 0.
    mod_data[9][np.where(mod_data[9] < -2.)] = 0.
    mod_data[10][np.where(mod_data[10] < -2.)] = 0.
    mod_data[13][np.where(mod_data[13] < -2.)] = 0.

    #Mask zeros 
    mod_data = [ma.masked_where(v == 0., v) for v in mod_data]
    
    del mods
    gc.collect()

    

