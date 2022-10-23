#!/usr/bin/env python
# coding: utf-8

import itertools
from collections import Counter
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import seaborn as sns


def plot_embedding_targets(X_embedded, y, alpha=1., palette=None):
    fig = sns.set(rc={'figure.figsize':(14,14)})
    
    if palette is None:
        cntr = Counter(y)
        palette = sns.color_palette("bright", len(cntr.keys()))

    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y, legend='full', palette=palette, alpha=alpha)
    
    return fig



def plot_classwise_dist(df, label_col):
    # get all unique labels
    cntr = Counter(df[label_col])
    cntr.keys()

    # we want a legend for mean and std for each class
    legend_entries = [[f"{item:.2f}_mean", f"{item:.2f}_std"] for item in list(cntr.keys())]

    # flatten legend entry list
    legend_entries = list(itertools.chain(*legend_entries))

    # create palette, so we can use the same color for the mean and std for each class
    palette = sns.color_palette("bright", len(cntr.keys()))

    for i, (key, group) in enumerate(df.groupby(label_col)):
        mean_spec = group.drop(columns=label_col).mean(axis=0)
        std_spec = group.drop(columns=label_col).std(axis=0)

        lower_bound = mean_spec-std_spec
        upper_bound = mean_spec+std_spec

        # plot mean values
        fig = sns.lineplot(x=range(len(mean_spec)), y=mean_spec, color=palette[i])
        # plot std values
        plt.fill_between(range(len(mean_spec)), lower_bound, upper_bound, color=palette[i], alpha=.1)

    plt.legend(legend_entries, ncol=3, loc='best', title=label_col)
    
    return fig


def plot_classwise_kde(df, label_col, feature_idx=0, focus=-1):
    focus_label = 'None'

    x = df.columns[feature_idx]
    # todo: do only once
    # get all unique labels
    cntr = Counter(df[label_col])
    labels = list(cntr.keys())

    # we want a legend with formatted strings
    legend_entries = [f"{item:.2f}" for item in list(cntr.keys())]
    palette = sns.color_palette("bright", len(cntr.keys()))

    
    if focus>=0:
        focus_label = labels[focus]

    for i, (key, group) in enumerate(df.groupby(label_col)):
        lw = 1
        # plot focused kde with thicker line
        if key == focus_label:
            lw = 5
        fig = sns.kdeplot(data=group, x=x,
                    linewidth=lw,
                    color=palette[i])
        
    plt.legend(legend_entries, ncol=3, loc='best', title=label_col)
    plt.title(f"feature: {x}, focus: {focus_label}");
    
    return fig
