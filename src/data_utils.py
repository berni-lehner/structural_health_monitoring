"""The data_utils module provides the Python interfaces to interact with data.
 Acknowledgements for inspiration goes to https://betterprogramming.pub/how-to-know-zip-content-without-downloading-it-87a5b30be20a

"""

__author__ = ("Bernhard Lehner <https://github.com/berni-lehner>")


import os
import pandas as pd
import numpy as np
from pathlib import Path

from LogFilterbank import LogFilterbank


# measurement characteristics are separated by underscores and encoded as follows:
# P_R0.0_0.000125_0.019125_1.005_1.0043_1.0443_1.0047_1.0338_0.02_.csv

FEATURE_LIST = ['y_cat', #P Pristine/D Defect: pristine structure (initial structure without any damage)/ damaged structure (sandwich structure with circular face layer debonding)
                'y_radius', #circular face layer debonding with a radial size of e.g., R_0.005 means 5mm)
                'y_sdv_fl', #structural damping value of the face layer material
                'y_sdv_core', #structural damping value of the core material (aluminum honeycomb)
                'y_dens_core', #density of the core material, deviation in %
                'y_young_fl', #Young's modulus of the face layer material, deviation in % (aluminum)
                'y_dc', #dielectric constant (PWAS/Sensor), deviation in % 
                'y_pcf', #piezoelectric coupling factors (PWAS/Sensor), deviation in %, same for all piezoelectric factors
                'y_ec_pwas', #elastic compliance of PWAS material (Sensor) deviation in %, same for all comliance values
                'y_loss_f', # loss factor (added after simulation)
               ]
    
def load_raw_data(file_name):
    X_df = pd.read_csv(file_name, header=None, delimiter=',', dtype=np.float32)
    X_df.columns = ['kHz', 'real', 'imag']

    # magnitude spectrum
    X_df['abs'] = np.abs(X_df['real'].values + 1j*X_df['imag'].values)
        
    # name w/o path
    f_name = os.path.split(file_name)[1]
    
    # naming convention: characteristics are separated by underscore
    file_parts = f_name.split('_')
    
    y_df = pd.DataFrame(columns=FEATURE_LIST)
    
    # categorical target
    y_df[FEATURE_LIST[0]] = pd.Series(file_parts[0], dtype='str')
        
    # numerical target: defect radius
    radius = np.float32(file_parts[1][1:]) #ignore R at beginning
    y_df[FEATURE_LIST[1]] = pd.Series(radius*1000, dtype='float32')
    
    # iterate the remaining characteristics
    offset = 2
    for i,feature in enumerate(FEATURE_LIST[offset:]):
        y_df[feature] = pd.Series(file_parts[i+offset], dtype='float32')
        
    return X_df, y_df


def get_log_spec(X, fb:LogFilterbank, to_dB=False):
    log_spec = fb.apply(spec=X, to_dB=to_dB)
    
    return log_spec


def load_processed_data(file_names,
                        fb:LogFilterbank,
                        X_col='real',
                        y_col=['y_cat'],
                        to_dB=False,
                        cache_file=None):
    df = None
    cached = False
    
    # try to load from given cache_file ...
    if cache_file is not None:
        if Path.exists(cache_file):
            cached = True
            df = pd.read_pickle(cache_file)
            
    # ... otherwise load from scratch
    if df is None:
        # str or list of strings
        if(type(file_names) == list):
            # load dataframes into list
            dfs = [_load_processed_data(file_name, fb, X_col, y_col, to_dB) for file_name in file_names]

            # concatenate into single dataframe
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = _load_processed_data(file_names, fb, X_col, y_col, to_dB)
    
    # convert entries to be compatible with interfaces like Counter, etc...
    # TODO: maybe there is a better way to do this in one step?
    if not cached:
        for col in y_col:
            df[col] = [y[0] for y in df[col].values]

    # cache if need be
    if cache_file is not None and not cached:
        df.to_pickle(cache_file)

    return df


def load_raw_specs(file_names,
                   X_col='real',
                   y_col=['y_cat'],
                   cache_file=None):
    df = None
    cached = False
    
    # try to load from given cache_file ...
    if cache_file is not None:
        if Path.exists(cache_file):
            cached = True
            df = pd.read_pickle(cache_file)
            
    # ... otherwise load from scratch
    if df is None:
        # str or list of strings
        if(type(file_names) == list):
            # load dataframes into list
            dfs = [_load_raw_specs(file_name, X_col, y_col) for file_name in file_names]

            # concatenate into single dataframe
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = _load_raw_specs(file_names, X_col, y_col)
            
    # convert entries to be compatible with interfaces like Counter, etc...
    # TODO: maybe there is a better way to do this in one step?
    if not cached:
        for col in y_col:
            df[col] = [y[0] for y in df[col].values]
    
    # cache if need be
    if cache_file is not None and not cached:
        df.to_pickle(cache_file)
    
    return df


def _load_raw_specs(file_name, X_col='real', y_col=['y_cat']):
    X_df, y_df = load_raw_data(file_name)

    # extract features
    spec = X_df[X_col].values
        
    # create new DataFrame containing features
    col_names = ['spec_' + str(i) for i in range(len(spec))]
    df = pd.DataFrame(columns=col_names, data=[spec])

    # combine with target variable(s)
    for col in y_col:
        df[col] = [y_df[col]]
        
    df['file'] = [file_name]
    
    return df


def _load_processed_data(file_name, fb:LogFilterbank, X_col='real', y_col=['y_cat'], to_dB=False):    
    X_df, y_df = load_raw_data(file_name)

    # extract features
    log_spec = get_log_spec(X=X_df[X_col].values, fb=fb, to_dB=to_dB)
        
    # create new DataFrame containing features
    col_names = ['logspec_' + str(i) for i in range(fb.n_log_bins)]
    df = pd.DataFrame(columns=col_names, data=[log_spec])

    # combine with target variable(s)
    for col in y_col:
        df[col] = [y_df[col]]
        
    df['file'] = [file_name]
    
    
    return df    