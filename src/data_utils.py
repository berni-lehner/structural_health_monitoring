"""The data_utils module provides the Python interfaces to interact with data.
"""
__author__ = ("Bernhard Lehner <https://github.com/berni-lehner>")


import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

from LogFilterbank import LogFilterbank
from feature_utils import extract_dctc



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

# measurement characteristics are separated by underscores:
REAL_FEATURE_LIST = ['y_cat', #P Pristine/D Defect: pristine structure (initial structure without any damage)/ damaged structure (sandwich structure with circular face layer debonding)
                'y_radius', #circular face layer debonding with a radial size in [mm])
                'interference', # normal, finger, bulge, etc...
                'variant', #additional info (1-4; optional)
                'comment', #additional info (oval, new, old; optional)
               ]


def init_data(syn_data_path=None, real_data_path=None):
    from DataDownloader import DataDownloader as ddl
    
    # synthetic data
    if syn_data_path is not None:
        url = r"https://sandbox.zenodo.org/record/1159057/files/data_synthetic.zip"

        start_time = time.perf_counter()
        dl_succeed = ddl.download_and_unpack(url, syn_data_path, cache=True)
        end_time = time.perf_counter()
        print(f"time passed: {end_time-start_time:.2f} s")
        print(f"downloading synthetic data successful: {dl_succeed}")
    
    # real world data
    if real_data_path is not None:
        url = r"https://sandbox.zenodo.org/record/1159057/files/real_world.zip"

        start_time = time.perf_counter()
        dl_succeed = ddl.download_and_unpack(url, real_data_path, cache=True)
        end_time = time.perf_counter()
        print(f"time passed: {end_time-start_time:.2f} s")
        print(f"downloading real world data successful: {dl_succeed}")
    
    
    

def load_raw_data(file_name):
    X_df = pd.read_csv(file_name, header=None, delimiter=',', dtype=np.float32)
    X_df.columns = ['kHz', 'real', 'imag']

    # magnitude spectrum
    #X_df['abs'] = np.abs(X_df['real'].values + 1j*X_df['imag'].values)
        
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


def load_real_data(file_name):
    X_df = pd.read_csv(file_name, header=None, delimiter=',', dtype=np.float32)
    X_df.columns = ['kHz', 'real', 'imag']

    # magnitude spectrum
    #X_df['abs'] = np.abs(X_df['real'].values + 1j*X_df['imag'].values)
    
    # name w/o path and extension
    f_name = file_name.stem if type(file_name) is Path else Path(file_name).stem
    
    # naming convention: characteristics are separated by underscore
    file_parts = f_name.split('_')
    
    radius = np.float32(file_parts[0])
    category = 'P' if radius==0 else 'D'

    variant = ''
    comment = ''
    
    if len(file_parts) > 2:
        fpart = file_parts[2]
        if fpart.isdigit():
            variant = fpart
        else:
            comment = fpart

        if len(file_parts) == 4:
            assert len(comment)==0
            comment = file_parts[3]
            
    y_cat = pd.Series(category, dtype='str')
    y_radius = pd.Series(radius, dtype='float32')
    interference = pd.Series(file_parts[1], dtype='str')
    variant = pd.Series(variant, dtype='str')
    comment = pd.Series(comment, dtype='str')

    y_df = pd.DataFrame(columns=REAL_FEATURE_LIST)
    
    for i,feature in enumerate(REAL_FEATURE_LIST):
        y_df[feature] = eval(feature)
         
    return X_df, y_df


def get_log_spec(X, fb:LogFilterbank, to_dB=False):
    log_spec = fb.apply(spec=X, to_dB=to_dB)
    
    return log_spec


def load_processed_data(file_names,
                        fb:LogFilterbank,
                        X_col='real',
                        y_col=['y_radius'],
                        to_dB=False,
                        cache_file=None,
                        synthetic=True,
                        calibration_file=None):
    df = None
    cached = False
    
    # try to load from given cache_file ...
    if cache_file is not None:
        if Path.exists(cache_file):
            cached = True
            df = pd.read_pickle(cache_file)
            
    # ... otherwise load from scratch
    if df is None:
        # calibration
        calibration = None
        if calibration_file is not None:
            calibration = pd.read_pickle(calibration_file).values

        # str or list of strings
        if(type(file_names) == list):
            # load dataframes into list
            dfs = [_load_processed_data(file_name, fb, X_col, y_col, to_dB, synthetic, calibration) for file_name in file_names]

            # concatenate into single dataframe
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = _load_processed_data(file_names, fb, X_col, y_col, to_dB, synthetic, calibration)
    
    # convert entries to be compatible with interfaces like Counter, etc...
    # TODO: maybe there is a better way to do this in one step?
    if not cached:
        for col in y_col:
            df[col] = [y[0] for y in df[col].values]

    # cache if need be
    if cache_file is not None and not cached:
        with open('path_to_file.pkl', 'rb') as f:
            df.to_pickle(cache_file, protocol=pickle.HIGHEST_PROTOCOL)

    return df


def _load_processed_data(file_name, fb:LogFilterbank, X_col='real',
                         y_col=['y_radius'], to_dB=False, synthetic=True,
                         calibration=None):
    if synthetic:
        X_df, y_df = load_raw_data(file_name)
    else:
        X_df, y_df = load_real_data(file_name)

    # extract features
    spec = X_df[X_col].values
    
    if calibration is not None:
        spec += calibration
 
    # extract features
    log_spec = get_log_spec(X=spec, fb=fb, to_dB=to_dB)
        
    # create new DataFrame containing features
    col_names = ['logspec_' + str(i) for i in range(fb.n_log_bins)]
    df = pd.DataFrame(columns=col_names, data=[log_spec])

    # combine with target variable(s)
    for col in y_col:
        df[col] = [y_df[col]]
        
    df['file'] = pd.Series([file_name], dtype="string")    
    
    return df


def load_raw_specs(file_names,
                   X_col='real',
                   y_col=['y_radius'],
                   synthetic=True,
                   cache_file=None,
                   calibration_file=None):
    df = None
    cached = False
    
    # try to load from given cache_file ...
    if cache_file is not None:
        if Path.exists(cache_file):
            cached = True
            df = pd.read_pickle(cache_file)
            
    # calibration
    calibration = None
    if calibration_file is not None:
        calibration = pd.read_pickle(calibration_file).values
            
    # ... otherwise load from scratch
    if df is None:
        # str or list of strings
        if(type(file_names) == list):
            # load dataframes into list
            dfs = [_load_raw_specs(file_name, X_col, y_col, synthetic, calibration) for file_name in file_names]

            # concatenate into single dataframe
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = _load_raw_specs(file_names, X_col, y_col, synthetic)
            
    # convert entries to be compatible with interfaces like Counter, etc...
    # TODO: maybe there is a better way to do this in one step?
    if not cached:
        for col in y_col:
            df[col] = [y[0] for y in df[col].values]
    
    # cache if need be
    if cache_file is not None and not cached:
        df.to_pickle(cache_file)
    
    return df


def _load_raw_specs(file_name, X_col='real', y_col=['y_radius'], synthetic=True, calibration=None):
    if synthetic:
        X_df, y_df = load_raw_data(file_name)
    else:
        X_df, y_df = load_real_data(file_name)

    # extract features
    spec = X_df[X_col].values
    
    if calibration is not None:
        spec += calibration
        
    # create new DataFrame containing features
    col_names = ['spec_' + str(i) for i in range(len(spec))]
    df = pd.DataFrame(columns=col_names, data=[spec])

    # combine with target variable(s)
    for col in y_col:
        df[col] = [y_df[col]]
        
    df['file'] = pd.Series([file_name], dtype="string")
    
    return df




def load_preprocessed_data(data_path, target_col=['y_radius'], synthetic=True, f_max=None, n_log_bins=87, calibration_file=None, cache=False):
    # configuration
    sr = 120000 #originally from df['kHz'].iloc[-1]*1000*2 # from measurement, highest f[kHz]*2
    #n_log_bins = 87
    n_fft = 1600
    n_fft_bins = 801
    f_min = 1300
    norm = 'height'

    fb = LogFilterbank(sr=sr, n_fft_bins=n_fft_bins, n_log_bins=n_log_bins, f_min=f_min, f_max=f_max, norm=norm)

    to_dB = True

    file_names = list(data_path.glob('**/*.csv'))
    print(f"files: {len(file_names)}")

    # cache file for faster data loading on later iterations
    pickle_name = None
    if cache:
        pickle_name = Path(data_path, f"filtered_specs__fmin_{fb.f_min}__fmax_{fb.f_max}__lbins_{fb.n_log_bins}.pkl")

    df = load_processed_data(file_names, fb,
#                             y_col=[FEATURE_LIST[0], FEATURE_LIST[1]],
                             y_col=target_col,
                             to_dB=to_dB,
                             synthetic=synthetic,
                             cache_file=pickle_name,
                             calibration_file=calibration_file)
    
    X = df[df.columns[0:fb.n_log_bins]].values

    n_coeffs = 32
    X_dct = extract_dctc(X=X, n_coeffs=n_coeffs)
    
    dct_df = pd.DataFrame(X_dct)
    dct_df[target_col] = df[target_col]
    
    return dct_df, fb



def load_syn_reg_data(data_path,
                      f_max=24000,
                      n_log_bins=66,
                      min_radius=2.5,
                      synthetic=True,
                      cache=False,
                      calibration_file=None):
    '''
    Load data for regression experiments
    '''

    target_col='y_radius'

    df, fb = load_preprocessed_data(data_path, target_col=[target_col],
                                    f_max=f_max, n_log_bins=n_log_bins,
                                    synthetic=synthetic,
                                    cache=cache,
                                    calibration_file=calibration_file)
    
    # round target values
    df[target_col] = df[target_col].astype(float).round(1)
    
    # drop very small radii
    df_normal =  df[df[target_col]==0.0]
    df_anomaly = df[df[target_col]>min_radius]

    Xpos = df_normal.drop(columns=[target_col]).values
    ypos = df_normal[target_col].values

    Xneg = df_anomaly.drop(columns=[target_col]).values
    yneg = df_anomaly[target_col].values
    
    # merge 0 size and other defect size data
    X = np.concatenate([Xpos, Xneg], axis=0)
    y = np.concatenate([ypos, yneg], axis=0)    
    
    # skip the 1st feature for stability
    return X[:,1:], y



def load_reg_data_base(data_path,
                       min_radius=2.5,
                       synthetic=True,
                       cache=False,
                       calibration_file=None):
    
    file_names = list(data_path.glob('**/*.csv'))

    # cache file for faster data loading on later iterations
    pickle_name = None
    if cache:
        pickle_name = Path(data_path, 'raw_specs.pkl')
        
    df = load_raw_specs(file_names=file_names,
                        synthetic=synthetic,
                        cache_file=pickle_name,
                        calibration_file=calibration_file)
    
    
    target_col = 'y_radius'
    df = df.drop(columns='file')

    # make sure the class labels are sorted for further convenience
    df = df.sort_values(by=target_col)
    
    df[target_col] = df[target_col].astype(float).round(1)
    
    # drop very small radii
    df_normal =  df[df[target_col]==0.0]
    df_anomaly = df[df[target_col]>min_radius]

    Xpos = df_normal.drop(columns=[target_col]).values
    ypos = df_normal[target_col].values

    Xneg = df_anomaly.drop(columns=[target_col]).values
    yneg = df_anomaly[target_col].values
    
    # merge 0 size and other defect size data
    X = np.concatenate([Xpos, Xneg], axis=0)
    y = np.concatenate([ypos, yneg], axis=0)    
    
    return X, y