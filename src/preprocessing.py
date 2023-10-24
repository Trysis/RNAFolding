#!/usr/bin/env python
# coding: utf-8


import re
import argparse
import pandas as pd
import numpy as np

from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Local modules
import auxiliary

# Constantes
RNA_BASES = [["A"], ["U"], ["C"], ["G"]]
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(RNA_BASES)

# Function to keep rows with maximum signal_to_noise within identical sequences
def filter_identical_sequences(df):
    # Group by 'sequence' and keep the row with max 'signal_to_noise'
    filtered_df = df.groupby('sequence').apply(lambda x: x.loc[x['signal_to_noise'].idxmax()])
    return filtered_df


def dict_from_data(data, keys_name=["2A3_MaP", "DMS_MaP"]):
    n_duplicate = sum(data.duplicated(subset=["sequence", "experiment_type"]))
    if n_duplicate > 0:
        return None

    seq_reactivity = dict()
    for seq, group in data.groupby("sequence"):
        seq_reactivity[seq] = dict()
        seq_reactivity[seq]["id"] = group["sequence_id"].values[0]
        for key in keys_name:
            mask = group["experiment_type"] == key
            seq_reactivity[seq][key] = group[mask].drop(
                labels=["sequence", "experiment_type"], axis=1
            ).values.reshape(-1)
            seq_reactivity[seq][key] = np.expand_dims(seq_reactivity[seq][key], axis=1)

    return seq_reactivity


def XY_from_dict(dict_data, encoder, maxlen=457):
    id_list = []
    x_list = []
    y_list = []
    for sequence, reactivities in dict_data.items():
        y = np.hstack([reactivities["2A3_MaP"], reactivities["DMS_MaP"]])
        x_list.append(onehot_from_sequence(sequence, encoder, maxlen=maxlen))
        y_list.append(padded_matrix(y, maxlen=maxlen))
        id_list.append(reactivities["id"])

    return np.array(x_list), np.array(y_list), np.array(id_list)


def onehot_from_sequence(sequence, encoder, to_add="0", maxlen=457):
    """sequence: str"""
    if maxlen is None:
        maxlen = 0
    proccessed_sequence = sequence.upper()
    proccessed_sequence += to_add * (maxlen - len(sequence))
    proccessed_sequence = [[nbase] for nbase in proccessed_sequence]
    onehot_sequence = encoder.transform(proccessed_sequence)
    return onehot_sequence


def padded_matrix(matrix_2d, maxlen=457):
    """"""
    if not isinstance(matrix_2d, np.ndarray):
        matrix_2d = np.array(matrix_2d)

    n_toadd = maxlen - matrix_2d.shape[0]
    padding = ((0, n_toadd), (0, 0))  # padding on axis
    matrix_2d_padded = np.pad(matrix_2d, pad_width=padding, mode="constant")
    return matrix_2d_padded


def reactivity_normalization(df_2A3_MaP, df_DMS_MaP, y_columns):
    """
    Robust Z-score Normalization of reactivities

    Parameters:
    - df_2A3_MaP(DataFrame): dataframe with the 2A3 experiment results
    - df_DMS_MaP(DataFrame): dataframe with the DMS experiment results

    Returns:
    - df_2A3_MaP(DataFrame): Normalized dataframe with the 2A3 experiment results
    - df_DMS_MaP(DataFrame) : Normalized dataframe with the DMS experiment results
    """
    # Calculate median and Median Absolute Deviation (MAD) for robust normalization
    a3_median = np.nanmedian(df_2A3_MaP[y_columns].values.reshape(-1))
    a3_mad = np.nanmedian(np.abs(df_2A3_MaP[y_columns].values.reshape(-1) - a3_median), axis=0)
    dms_median = np.nanmedian(df_DMS_MaP[y_columns].values.reshape(-1))
    dms_mad = np.nanmedian(np.abs(df_DMS_MaP[y_columns].values.reshape(-1) - dms_median), axis=0)

    # Apply robust z-score normalization to all DataFrames from column 5 onwards
    cst = 1.482602218505602
    df_2A3_MaP[y_columns] = (df_2A3_MaP[y_columns] - a3_median) / (cst * a3_mad)
    df_DMS_MaP[y_columns] = (df_DMS_MaP[y_columns] - dms_median) / (cst * dms_mad)

    return df_2A3_MaP, df_DMS_MaP


if __name__ == "__main__":
    data = pd.read_csv("./data/SN_filtered_train.csv", header=0)

    # Get columns name for X, and Y
    x_columns = ["sequence_id", "sequence"]
    conditional_columns = ["experiment_type", "signal_to_noise"]
    y_columns = [colname for colname in data.columns if re.match("^reactivity_[0-9]{4}$", colname)]

    # Keep the necessary columns from the DataFrame
    cleaned_data = data[x_columns + conditional_columns + y_columns]

    # Create two separate DataFrames based on "experiment_type"
    df_2A3_MaP = cleaned_data[cleaned_data['experiment_type'] == '2A3_MaP']
    df_DMS_MaP = cleaned_data[cleaned_data['experiment_type'] == 'DMS_MaP']

    # Delete cleaned_train_data to free space memory
    del cleaned_data

    df_2A3_MaP_filtered = filter_identical_sequences(df_2A3_MaP)  # Filter df_2A3_MaP
    df_DMS_MaP_filtered = filter_identical_sequences(df_DMS_MaP)  # Filter df_DMS_MaP

    df_2A3_MaP_normalized, df_DMS_MaP_normalized = reactivity_normalization(df_2A3_MaP_filtered, df_DMS_MaP_filtered, y_columns)

    # Concatenate the two data frames
    mask_2A3 = df_2A3_MaP_normalized["sequence"].isin(df_DMS_MaP_normalized["sequence"])
    mask_DMS = df_DMS_MaP_normalized["sequence"].isin(df_2A3_MaP_normalized["sequence"])

    cleared_data = pd.concat([df_2A3_MaP_normalized[mask_2A3], df_DMS_MaP_normalized[mask_DMS]], ignore_index=True)
    cleared_data.drop(columns=['signal_to_noise'], inplace=True)

    # columns type
    cleared_data[y_columns] = cleared_data[y_columns].astype(np.float32)

    # Save cleared_train_data as a CSV file
    csv_path = './data/cleared_data.csv'
    cleared_data.to_csv(csv_path, index=False)

    dict_data = dict_from_data(cleared_data)
    x, y, id = XY_from_dict(dict_data, encoder)

    # Split the data into training and validation sets
    x_train, x_val, x_test, \
    y_train, y_val, y_test, \
    id_train, id_val, id_test = \
        auxiliary.train_val_test_split(x, y, id)

    auxiliary.save_npy(
        x, "x.npy",
        y, "y.npy",
        id, "id.npy",
        x_train, "x_train.npy",
        y_train, "y_train.npy",
        id_train, "id_train.npy",
        x_val, "x_val.npy",
        y_val, "y_val.npy",
        id_val, "id_val.npy",
        x_test, "x_test.npy",
        y_test, "y_test.npy",
        id_test, "id_test.npy"
    )