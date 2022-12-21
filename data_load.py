import pandas as pd
import numpy as np
import torch
from datacentertracesdatasets import loadtraces
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def get_ori_data(sequence_length=60, stride=1, shuffle=False, seed=13, trace='alibaba2018'):
    np.random.seed(seed)
    original_df = loadtraces.get_trace(trace_name=trace, trace_type='machine_usage', stride_seconds=300,
                                       format='dataframe')
    start_sequence_range = list(range(0, original_df.shape[0] - sequence_length, stride))
    if shuffle:
        np.random.shuffle(start_sequence_range)
    batched_original_x = np.array([original_df[start_index:start_index+sequence_length] for start_index in start_sequence_range])
    batched_original_y = np.ones(batched_original_x.shape[0])
    return batched_original_x, batched_original_y

def scale_data(ori_data, scaling_method='standard', scaler=None):
    reshaped_ori_data = ori_data.reshape(-1, 1)
    if scaler is None:
        assert scaling_method in ['standard', 'minmax'], 'Only standard and minmax scalers are currently supported'
        if scaling_method == 'minmax':
            scaler = MinMaxScaler(feature_range=(-1, 1))
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
