#!/usr/bin/env python
# coding: utf-8

import itertools
from collections import Counter
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns

# general plot configuration
SMALL_SIZE = 10
MEDIUM_SIZE = 16
LARGE_SIZE = 20
HUGE_SIZE = 24
FIG_SIZE = (22, 8)

def init_plotting():
    # Matplotlib
    plt.rc('figure', figsize=FIG_SIZE)        # default figure size
    plt.rc('figure', titlesize=HUGE_SIZE)     # fontsize of the figure title
    plt.rc('figure', titleweight='bold')      # weight of the figure title
    #plt.rc('font', size=MEDIUM_SIZE)          # default text sizes
    #plt.rc('axes', titlesize=LARGE_SIZE)      # fontsize of the axes title
    #plt.rc('axes', titleweight='bold')        # weight of the axes title
    #plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
    #plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    #plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('legend', title_fontsize=MEDIUM_SIZE)    # legend fontsize
    
    # Seaborn
    sns.set(rc={"figure.figsize": FIG_SIZE,
                "figure.titlesize": HUGE_SIZE,
                "figure.titleweight": 'bold',
    #            "font.size": MEDIUM_SIZE,
    #            "axes.titlesize": LARGE_SIZE,
    #            "axes.titleweight": 'bold',
    #            "axes.labelsize": MEDIUM_SIZE,
    #            "xtick.labelsize": MEDIUM_SIZE,
    #            "ytick.labelsize": MEDIUM_SIZE,
    #            "legend.fontsize": MEDIUM_SIZE,
                "legend.title_fontsize": MEDIUM_SIZE,
               })    

    
def plot_embedding_targets(X_embedded, y, alpha=1., palette=None):
    fig = plt.figure()
    
    cntr = Counter(y)
    if palette is None:
        palette = sns.color_palette("bright", len(cntr.keys()))

    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y,
                    legend='full', palette=palette, alpha=alpha)
    
    return fig



def plot_classwise_dist(df, label_col=None, palette=None):
    fig = plt.figure()
    
    # last column contains the labels if not handled otherwise
    if label_col is None:
        label_col = df.columns[-1]
        
    # get all unique labels
    cntr = Counter(df[label_col])

    if palette is None:
        palette = sns.color_palette("bright", len(cntr.keys()))

    for i, (key, group) in enumerate(df.groupby(label_col)):
        # plot mean values
        mean_spec = group.drop(columns=label_col).mean(axis=0)
        sns.lineplot(x=range(len(mean_spec)), y=mean_spec, color=palette[i],
                     label=f"{key:.2f}_mean")

        # plot std values
        std_spec = group.drop(columns=label_col).std(axis=0)
        lower_bound = mean_spec-std_spec
        upper_bound = mean_spec+std_spec
        plt.fill_between(range(len(mean_spec)), lower_bound, upper_bound,
                         color=palette[i], alpha=.1, label=f"{key:.2f}_std")
    
    return fig


def plot_classwise_kde(df, label_col, labels, palette=None, feature_idx=0, focus=-1, focus_lw=5):    
    fig = plt.figure()

    focus_label = 'None'

    x = df.columns[feature_idx]
    
    if focus>=0:
        focus_label = labels[focus]

    if palette is None:
        # get all unique labels
        cntr = Counter(df[label_col])
        palette = sns.color_palette("bright", len(cntr.keys()))

    for i, (key, group) in enumerate(df.groupby(label_col)):
        lw = None # default linewidth
        # plot focused kde with thicker line
        if key == focus_label:
            lw = focus_lw # focused linewidth
        
        sns.kdeplot(data=group, x=x, linewidth=lw, color=palette[i],
                          label=f"{key:.2f}")
        
    plt.legend()
    
    return fig