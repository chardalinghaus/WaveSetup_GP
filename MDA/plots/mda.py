#!/usr/bin/env python
# -*- coding: utf-8 -*-

#MDA Plot original from HYWAVES (https://github.com/ripolln/hywaves)
#Adapted for the article's second figure.

# pip
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# import constants
from .config import _faspect, _fsize, _fdpi
import pandas as pd  #ADDED

def axplot_scatter_mda_vs_data(ax, x_mda, y_mda, x_data, y_data):
    'axes scatter plot variable1 vs variable2 mda vs data'

    # full dataset 
    ax.scatter(
        x_data, y_data,
        s = 25,
        facecolors = 'none',
        edgecolors = 'k',
        # marker = '.',
        # c = 'k',
        label = 'Testing Set'
    )
    
    # mda selection
    ax.scatter(
        x_mda, y_mda,
        marker = '+',
        c = 'r',
        s = 15,
        label='Training Set'
    )
    
def Plot_MDA_Data(pd_data, pd_mda, show=True):
    '''
    Plot scatter with MDA selection vs original data

    pd_data - pandas.DataFrame, complete data
    pd_mda - pandas.DataFrame, mda selected data

    pd_data and pd_mda should share columns names
    '''

    # TODO: activate dlab ? 

    ## figure conf.
    #d_lab = {
    #    'pressure_min': 'Pmin (mbar)',
    #    'gamma': 'gamma (º)',
    #    'delta': 'delta (º)',
    #    'velocity_mean': 'Vmean (km/h)',
    #}

    # variables to plot
    vns = pd_data.columns
    labels= pd.DataFrame (columns = [r'$\mathbf{\bar{\eta}_M}$ (m)', r'$\mathbf{H_{s0}}$ (m)', r'$\mathbf{T_p}$ (s)', r'$\mathbf{\beta_f}$', r'$\mathbf{L_0}$ (m)', r'$\mathbf{\xi_0}$', r'$\mathbf{D_{50}}$ (m)']).columns   ##added

    # filter
    vfs = ['n_sim']
    vns = [v for v in vns if v not in vfs]
    n = len(vns)

    # figure
    fig = plt.figure(figsize=(_faspect*_fsize, _faspect*_fsize))
    gs = gridspec.GridSpec(n-1, n-1, wspace=0.25, hspace=0.25)    #CHANGED (0.2 to 0.25)

    for i in range(n):
        for j in range(i+1, n):

            # get variables to plot
            vn1 = vns[i]
            vn2 = vns[j]

            # mda and entire-dataset
            vv1_mda = pd_mda[vn1].values[:]
            vv2_mda = pd_mda[vn2].values[:]

            vv1_dat = pd_data[vn1].values[:]
            vv2_dat = pd_data[vn2].values[:]

            # scatter plot 
            ax = plt.subplot(gs[i, j-1])
            # ax.tick_params(axis='y', rotation=90)   #Rotate Y axis  ##added
            axplot_scatter_mda_vs_data(ax, vv2_mda, vv1_mda, vv2_dat, vv1_dat)
            ax.tick_params(labelsize= 13)

            # custom axes
            labels1=labels[i]    #ADDED
            labels2=labels[j]    #ADDED
            
            if j==i+1:
                ax.set_xlabel(
                    #d_lab[vn2],
                    # vn2,    #CHANGED
                    labels2,    #CHANGED
                    {'fontsize':13, 'fontweight':'bold'} 
                )
            if j==i+1:
                ax.set_ylabel(
                    #d_lab[vn1],
                    # vn1,    #CHANGED
                    labels1,    #CHANGED
                    {'fontsize':13, 'fontweight':'bold'}
                )

            if i==0 and j==n-1:
                ax.legend(fontsize = 13, markerscale = 1.5)    #CHANGED
                
    # show and return figure
    if show: plt.show()
    return fig

