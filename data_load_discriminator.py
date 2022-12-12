import pandas as pd
import numpy as np
import torch
from datacentertracesdatasets import loadtraces
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def get_ori_data(sequence_length=60, stride=1, shuffle=False, seed=13, synthetic_data_filename=None, trace='azure_v2', label=1):
    np.random.seed(seed)
    original_df = loadtraces.get_trace(trace_name=trace, trace_type='machine_usage', stride_seconds=300,
                                       format='dataframe')
    start_sequence_range = list(range(0, original_df.shape[0] - sequence_length, stride))
    if shuffle:
        np.random.shuffle(start_sequence_range)
    batched_original_x = np.array([original_df[start_index:start_index+sequence_length] for start_index in start_sequence_range])
    batched_original_y = np.ones(batched_original_x.shape[0])

    synthetic_df = pd.read_csv(synthetic_data_filename, header=None)
    start_sequence_range = list(range(0, synthetic_df.shape[0] - sequence_length, stride))
    if shuffle:
        np.random.shuffle(start_sequence_range)
    batched_synthetic_x = np.array([synthetic_df[start_index:start_index+sequence_length] for start_index in start_sequence_range])
    batched_synthetic_y = np.zeros(batched_synthetic_x.shape[0])

    batched_x = np.concatenate((batched_original_x, batched_synthetic_x))
    batched_y = np.concatenate((batched_original_y, batched_synthetic_y))
    shuffled_x, shuffled_y = unison_shuffled_copies(batched_x, batched_y)
    return shuffled_x, shuffled_y


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def scale_data(ori_data, scaling_method='standard', scaler=None):
    reshaped_ori_data = ori_data.reshape(-1, 1)
    if scaler is None:
        assert scaling_method in ['standard', 'minmax'], 'Only standard and minmax scalers are currently supported'
        if scaling_method == 'minmax':
            scaler = MinMaxScaler()
            scaler.fit(reshaped_ori_data)
            scaler_params = [scaler.data_min_, scaler.data_max_]

        elif scaling_method == 'standard':
            scaler = StandardScaler()
            scaler.fit(reshaped_ori_data)
            scaler_params = [scaler.mean_, scaler.var_]
        scaled_ori_data = scaler.transform(reshaped_ori_data).reshape(ori_data.shape)
        return scaled_ori_data, scaler, scaler_params
    else:
        scaled_ori_data = scaler.transform(reshaped_ori_data).reshape(ori_data.shape)
        return scaled_ori_data, scaler, None
