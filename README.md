Code for the analysis and figures in Rugenstein, M., S. Dhame, D. Olonscheck, R. J. Wills, M. Watanabe & R. Seager, "Connecting the pattern problem and hot model problem"

The scripts used preprocessed large ensemble climate data which can be downloaded from https://esgf-data.dkrz.de/projects/esgf-dkrz/, https://esgf-data.dkrz.de/projects/cmip6-dkrz/ and https://www.earthsystemgrid.org/dataset/. The observational data sets can
be downloaded at HadISST1: https://www.metoffice.gov.uk/hadobs/hadisst/data/download.html, COBE: http://psl.noaa.gov/data/gridded/data.cobe.html, ERSSTv5 : https://www.ncei.noaa.gov/products/extended-reconstructed-sst. The effective climate sensitivity values are available through (Zelinka et al. 2020) and their updates on https://github.com/mzelinka/cmip56_forcing_feedback_ecs.

Method:
------
- All datasets were interpolated onto 2.5 x 2.5 global grid using the bilinear interpolation method of Climate Data Operator (CDO)
- Run calc_phi.py to store calculations of trends for the various time periods (from 1950 until 2021 in observations and models) and their correlation with the Effective Climate Sensitivity (EffCS) to dictionary (klepto)
- Run individual scripts for figures
