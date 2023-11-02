#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__title__ = "Correlation between phi and effective climate sensitivity"
__author__ = "Shreya Dhame"
__version__ = "3.6.3"
__email__ = "shreyadhame@gmail.com"

#============================================================
## Load modules
import os
import sys
from matplotlib.lines import Line2D

## My modules
from func import *
from load_le_data import *

#============================================================
### Execute script

if __name__ == "__main__":

    #Define time
    years=np.unique(time.dt.year)

    #Extract data from klepto
    db = klepto.archives.dir_archive('mphi',serialized=True,cached=False)
    ew_obs = db['ew_obs']
    ew_mods = db['ew_mods']
    ew_trends = db['ew_trends']
    ew_mtrends = db['ew_mtrends']
    ew_mphi = db['ew_mphi']
    mti = db['mti']

    #Pick models 
    mrc = ew_mods[1]
    can = ew_mods[-1]

    #Calculate trends
    #model 1
    ind1 = np.where(time.dt.year==1970)[0][0] #Index of start year
    ind2 = np.where(time.dt.year==1995)[0][0] #Index of end year
    chunk = ew_obs[ind1:ind2] #Select a chunk
    obs_trendl = mk_test(chunk)[-1]*10 #len(chunk)) or 10 for decadal trend

    mrc_trendsl = []
    for n in range(len(mrc)):
        chunk = mrc[n,ind1:ind2] #Select a chunk
        mrc_trendsl.append(mk_test(chunk)[-1]*10) #len(chunk)) or 10 for decadal trend
    can_trendsl = []
    for n in range(len(can)):
        chunk = can[n,ind1:ind2] #Select a chunk
        can_trendsl.append(mk_test(chunk)[-1]*10) #len(chunk)) or 10 for decadal trend

    #model2
    ind1 = np.where(time.dt.year==1990)[0][0] #Index of start year
    ind2 = np.where(time.dt.year==2015)[0][0] #Index of end year
    chunk = ew_obs[ind1:ind2] #Select a chunk
    obs_trendr = mk_test(chunk)[-1]*10 #len(chunk)) or 10 for decadal trend

    mrc_trendsr = []
    for n in range(len(mrc)):
        chunk = mrc[n,ind1:ind2] #Select a chunk
        mrc_trendsr.append(mk_test(chunk)[-1]*10) #len(chunk)) or 10 for decadal trend
    can_trendsr = []
    for n in range(len(can)):
        chunk = can[n,ind1:ind2] #Select a chunk
        can_trendsr.append(mk_test(chunk)[-1]*10) #len(chunk)) or 10 for decadal trend
    
    #create dataframes
    dfl = pd.DataFrame({'MIROC6':mrc_trendsl,'CanESM5':can_trendsl})
    dfl = pd.DataFrame({'MIROC6':mrc_trendsl,'CanESM5':can_trendsl})

    dfr = pd.DataFrame({'MIROC6':mrc_trendsr,'CanESM5':can_trendsr})
    dfr = pd.DataFrame({'MIROC6':mrc_trendsr,'CanESM5':can_trendsr})

    ## PLOT 
    fig = plt.figure(figsize=(25,6))
    widths = [1,1.5,1.5,1]
    heights = [1]
    spec = gridspec.GridSpec(ncols=4, nrows=1, width_ratios=widths,
                              height_ratios=heights,hspace=0.,wspace=0.05)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[0, 1])
    ax3 = fig.add_subplot(spec[0, 2])
    ax4 = fig.add_subplot(spec[0, 3])
    
    x = dfl['MIROC6']
    y = dfl['CanESM5']
    sns.histplot(x,stat='count', kde=False, fill=True,linewidth=0.,alpha=0.3,color='#E66100',ax=ax1,kde_kws={'cut':2})
    
    def normal(mean, std, histmax=False, color='#E66100'):
        x = np.linspace(mean-4*std, mean+4*std, 200)
        p = norm.pdf(x, mean, std)
        if histmax:
            p = p*histmax/max(p)
        z = ax1.plot(x, p, color, linewidth=2)
    normal(x.mean(), x.std(), histmax=ax1.get_ylim()[1])
    
    # sns.kdeplot(x,color='#E66100',ax=ax1,linewidth=2)
    sns.histplot(y,stat='count', kde=False, fill=True, linewidth=0.,alpha=0.3,color='#5D3A9B',ax=ax1,kde_kws={'cut':2})
    
    def normal(mean, std, histmax=False, color='#5D3A9B'):
        x = np.linspace(mean-4*std, mean+4*std, 200)
        p = norm.pdf(x, mean, std)
        if histmax:
            p = p*histmax/max(p)
        z = ax1.plot(x, p, color, linewidth=2)
    normal(y.mean(), y.std(), histmax=ax1.get_ylim()[1])
    
    # sns.kdeplot(y,color='#5D3A9B',ax=ax1,linewidth=2)
    ax1.axvline(obs_trendl,color='k',linewidth=2)
    ax1.set_xlabel('SST [$^\circ$C decade$^{-1}$]')
    ax1.set_title('(a)',loc='center')
    
    #Negative x-axis plot
    period = '1970-1995'
    ind = np.where(mti==period)[-1][0]
    yv = np.array(ecs)
    xv = ew_mphi[:,ind]
    
    ax2.scatter(
        xv, yv, color="#E1BE6A",
        s=80, alpha=0.8, zorder=10
    );
    
    X = xv.reshape(-1, 1)
    y = yv
    
    # Initialize linear regression object
    linear_regressor = LinearRegression()
    
    # Initialize linear regression object
    linear_regressor = LinearRegression()
    
    # Fit linear regression model of HDI on the log of CPI
    reg = linear_regressor.fit(X, y)
    print(reg.score(X,y))
    
    # Make predictions
    # * Construct a sequence of values
    x_pred = np.linspace(np.nanmin(xv).round(2), np.nanmax(xv).round(2), num=200).reshape(-1, 1)
    
    # * Use .predict() method with the created sequence
    y_pred = linear_regressor.predict(x_pred)  
    
    # Plot regression  line.
    # * Logarithmic transformation is reverted by using the exponential one.
    ax2.plot(x_pred, y_pred, color="#9897A9", lw=4, alpha=0.7)
    
    # Set default font size to 16
    plt.rcParams.update({"font.size": "16"})
    
    # Remove tick marks on both x and y axes
    ax2.yaxis.set_tick_params(length=0)
    ax2.set_yticks([])
    
    # Add grid lines, only for y axis
    ax2.grid(axis="y")
    
    # Remove all spines but keep the bottom one
    ax2.spines["left"].set_color("none")
    ax2.spines["right"].set_color("none")
    ax2.spines["top"].set_color("none")
    
    # Add labels -----------------------------------------------------
    mod_names = ['GFDL-ESM2M', 'MIROC6', 'MIROC-ES2L', 'GISS-E2.1-G', 'CESM1', 'NorCPM1', 'MPI-ESM', 'CanESM2', 'ACCESS-ESM1.5',\
                 'GFDL-CM3', 'CSIRO-Mk3.6', 'IPSL-CM6A-LR', 'CNRM-CM6.1', 'CESM2', 'CanESM5']
    
    TEXTS = []
    for idx, model in enumerate(mod_names):
        # Only append selected countries
        x, y = xv[idx], yv[idx]
        TEXTS.append(ax2.text(x, y, model, fontsize=12));
    
    # Adjust text position and add lines -----------------------------
    # 'expand_points' is a tuple with two multipliers by which to expand
    # the bounding box of texts when repelling them from points
    
    # 'arrowprops' indicates all the properties we want for the arrows
    # arrowstyle="-" means the arrow does not have a head (it's just a line!)
    adjust_text(
        TEXTS, 
        expand_points=(2, 2),
        arrowprops=dict(arrowstyle="-", lw=0.3),
        ax=ax2
    );
    
    #set title
    ax2.set_title('(b)',loc='center')
    
    # Add text for R2 and slope and period
    # ax2.text(-4.6,2.5,'R$^2$ = '+str(np.round(ew_r[ind],2)))
    ax2.text(-4.2,2.5,'R$^2$ = '+str(np.round(ew_r[ind],2)))
    ax2.text(-4.8,6.2,'1970 - 1995',fontsize=20)

    #Positive x-axis plot 
    period = '1990-2015'
    ind = np.where(mti==period)[-1][0]
    yv = np.array(ecs)
    xv = ew_mphi[:,ind]
    
    ax3.scatter(
        xv, yv, color="#40B0A6",
        s=80, alpha=0.8, zorder=10
    );
    
    X = xv.reshape(-1, 1)
    y = yv
    
    # Initialize linear regression object
    linear_regressor = LinearRegression()
    
    # Fit linear regression model of HDI on the log of CPI
    reg = linear_regressor.fit(X, y)
    print(reg.score(X,y))
    
    # Make predictions
    # * Construct a sequence of values
    x_pred = np.linspace(np.nanmin(xv).round(2), np.nanmax(xv).round(2), num=200).reshape(-1, 1)
    
    # * Use .predict() method with the created sequence
    y_pred = linear_regressor.predict(x_pred)  
    
    # Plot regression  line.
    # * Logarithmic transformation is reverted by using the exponential one.
    ax3.plot(x_pred, y_pred, color="#9897A9", lw=4, alpha = 0.7)
    
    # Remove tick marks on both x and y axes
    # ax3.xaxis.set_tick_params(length=0)
    ax3.tick_params(axis="y",direction="in", pad=-15)
    
    # Add grid lines, only for y axis
    ax3.grid(axis="y")
    
    # Remove all spines but keep the bottom one
    ax3.spines["left"].set_color("none")
    ax3.spines["right"].set_color("none")
    ax3.spines["top"].set_color("none")
    
    # Add labels -----------------------------------------------------
    mod_names = ['GFDL-ESM2M', 'MIROC6', 'MIROC-ES2L', 'GISS-E2.1-G', 'CESM1', 'NorCPM1', 'MPI-ESM', 'CanESM2', 'ACCESS-ESM1.5',\
                 'GFDL-CM3', 'CSIRO-Mk3.6', 'IPSL-CM6A-LR', 'CNRM-CM6.1', 'CESM2', 'CanESM5']
    
    TEXTS = []
    for idx, model in enumerate(mod_names):
        # Only append selected countries
        x, y = xv[idx], yv[idx]
        TEXTS.append(ax3.text(x, y, model, fontsize=12));
    
    # Adjust text position and add lines -----------------------------
    # 'expand_points' is a tuple with two multipliers by which to expand
    # the bounding box of texts when repelling them from points
    
    # 'arrowprops' indicates all the properties we want for the arrows
    # arrowstyle="-" means the arrow does not have a head (it's just a line!)
    adjust_text(
        TEXTS, 
        expand_points=(1.5, 1.5),
        arrowprops=dict(arrowstyle="-", lw=0.3),
        ax=ax3
    );
    
    # And finally set labels
    ax3.text(0.502,0.01,"\u03A6 [\u03C3]",transform=plt.gcf().transFigure,fontsize=16)
    ax3.set_ylabel("Effective Climate Sensitivity [K]",loc='top')
    # ax3.yaxis.set_label_position("right")
    
    ax3.set_title('(c)',loc='center')
    
    x = dfr['MIROC6']
    y = dfr['CanESM5']
    sns.histplot(x,stat='count', kde=False, fill=True,linewidth=0.,alpha=0.3,color='#E66100',ax=ax4,kde_kws={'cut':2})
    # sns.kdeplot(x,color='#E66100',ax=ax4,linewidth=2)
    
    def normal(mean, std, histmax=False, color='#E66100'):
        x = np.linspace(mean-4*std, mean+4*std, 200)
        p = norm.pdf(x, mean, std)
        if histmax:
            p = p*histmax/max(p)
        z = ax4.plot(x, p, color, linewidth=2)
    normal(x.mean(), x.std(), histmax=ax4.get_ylim()[1])
    
    
    sns.histplot(y,stat='count', kde=False, fill=True, linewidth=0.,alpha=0.3,color='#5D3A9B',ax=ax4,kde_kws={'cut':2})
    # sns.kdeplot(y,color='#5D3A9B',ax=ax4,linewidth=2)
    
    def normal(mean, std, histmax=False, color='#5D3A9B'):
        x = np.linspace(mean-4*std, mean+4*std, 200)
        p = norm.pdf(x, mean, std)
        if histmax:
            p = p*histmax/max(p)
        z = ax4.plot(x, p, color, linewidth=2)
    normal(y.mean(), y.std(), histmax=ax4.get_ylim()[1])
    
    ax4.axvline(obs_trendr,color='k',linewidth=2)
    ax4.set_xlabel('SST [$^\circ$C decade$^{-1}$]')
    
    colors = ['k','#E66100','#5D3A9B']
    # colors = ['k','#40B0A6','#E66100','#5D3A9B']
    title = ['Observations','MIROC6 | \nLow sensitivity model','CanESM5 | Hot model']
    # title = ['Observations','GFDL-ESM2M','MIROC-ES2L','CSIRO-Mk3.6']
    lines = [Line2D([0], [0], color=colors[i], linewidth=3) for i in range(len(colors))]
    ax4.legend(lines, title,loc='upper right',labelspacing=0.3,frameon=False)
    ax1.spines[['right', 'top']].set_visible(False)
    ax4.spines[['left', 'top']].set_visible(False)
    
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.set_ylim(0,18)
    ax1.set_ylim(0,18)
    
    #set axis limit
    ax3.set_xlim(0.,4.5)
    ax2.set_xlim(-4.5,-0.)
    
    ax1.grid(False)
    ax4.grid(False)
    
    # Add text for R2 and slope and period
    ax3.text(3.5,2.5,'R$^2$ = '+str(np.round(ew_r[ind],2)))
    ax3.text(3.8,6.2,'1990 - 2015',fontsize=20)
    
    ax4.set_title('(d)',loc='center')
    
    fig.set_facecolor("white") # set figure background color to white
    plt.savefig('Fig2.png', dpi=300, facecolor='white', bbox_inches="tight")
