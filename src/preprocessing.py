#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Reshape
from sklearn.model_selection import train_test_split

# Local modules
import auxiliary

# Constantes
RNA_BASES = [["A"], ["U"], ["C"], ["G"]]
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(RNA_BASES)


def load_data(filepath, **kwargs) :
    """Load data from a CSV file.

    Parameters:
        - filepath (str):
            The path to the CSV file containing the data.
        - kwargs (dict):
            Dictionnary containings optional parameters
            for pandas.read_csv function
    Returns:
        - data (pandas.DataFrame):
            A Pandas DataFrame containing the loaded data.

    """
    if not auxiliary.isfile(filepath):
        raise Exception(f"filepath is not a valid path.")

    data = pd.read_csv(filepath, **kwargs)
    return data


def save_data(cleared_data, filepath, **kwargs) :
    """Export the dataframe into a csv file
    
    Parameters : 
        - cleared_train_data(DataFrame) : dataframe with the whole data

    """
    cleared_train_data.to_csv(filepath, index=False, **kwargs)


def filter_identical_sequences(
    data,
    group_column=["sequence", "experiment_type"],
    signal_column="signal_to_noise"
):
    """For identical sequences keep rows
    with maximum signal to noise.

    Parameters:
        - data (DataFrame):
            pandas input dataframe

    Returns:
        - filtered_df (Dataframe):
            a filtered dataframe with the sequences
            with a maximum signal to noise.
    """
    filtered_df = cleaned_train_data.groupby(group_column).apply(
        lambda x: x.loc[x[signal_column].idxmax()]
    )
    return filtered_df


def filter_SN(data, sn_column="SN_filter"):
    """Filter the rows where "SN_filter" is equal to 1.

    Parameters:
        - data (DataFrame):
            A pandas Dataframe containing the input data

    Returns:
        - filtered_df (DataFrame):
            A dataframe containing only the sequences that
            passed the SN_filter.
    """
    filtered_df = data[data[sn_column] == 1]
    return filtered_df


def onehot_from_sequence(sequence, encoder, to_add="0", maxlen=457):
    """Takes a sequence and returns its one hot encoding.

    sequence: str
        RNA sequence, should be in upper character

    encoder: sklearn.preprocessing.OneHotEncoder
        Encoder to convert the sequence

    to_add:
        character to append until padding is reached
        with {maxlen}.

    maxlen: int
        Sequence maximum length for padding

    Returns: numpy.ndarray
        One hot encoding of the sequence.

    """
    if not maxlen:
        maxlen = 0

    proccessed_sequence = sequence.upper()
    proccessed_sequence += to_add * (maxlen - len(sequence))
    proccessed_sequence = [[nbase] for nbase in proccessed_sequence]
    onehot_sequence = encoder.transform(proccessed_sequence)

    return onehot_sequence


def encode_sequences(sequences, encoder, to_add="0", maxlen=457):
    """Returns the set of encoded sequences for a list of sequences."""
    enc_list = []
    for sequence in sequences:
        enc_list.append(onehot_from_sequence(sequence, encoder, to_add, maxlen))

    return np.array(enc_list)


def pad_matrix(matrix_2d, maxlen=457):
    """Takes a 2D matrix and add padding.

    matrix_2d: numpy.ndarray
        2D numpy array matrix of shape (n, m)

    maxlen: int
        {m} maximum length for padding

    Returns: numpy.ndarray
        The padded matrix if {m}<{maxlen} else it
        returns the matrix.

    """
    if not isinstance(matrix_2d, np.ndarray):
        matrix_2d = np.array(matrix_2d)

    if maxlen - matrix_2d.shape[0] <= 0:
        return matrix_2d

    add_len = maxlen - matrix_2d.shape[0]
    padding = ((0, add_len), (0, 0))  # padding on axis
    matrix_2d_padded = np.pad(matrix_2d, pad_width=padding, mode="constant")

    return matrix_2d_padded


def pad_matrices(matrices, maxlen=457):
    """Takes a set of 2D matrix and add padding to each of them."""
    matrices_list = []
    for matrix in matrices:
        matrices_list.append(pad_matrix(matrix, maxlen))

    return np.array(matrices_list)


def get_x(data, colname=["sequence"], dtype=np.float32):
    if isinstance(colname, str):
        colname = [colname]
    elif not isinstance(colname, list):
        colname = list(colname)


    
def get_target(cleared_train_data, to_match="^reactivity_[0-9]{4}$", dtype=np.float32) :
    """Extract reactivity columns as targets to use as Y.

    Parameters:
        cleared_train_data(DataFrame):
            pandas dataframe with the whole data
        to_match: str
            Pattern to match with the columns
        dtype:
            Type to convert target into

    Returns:
        targets(DataFrame):
            dataframe with reactivity columns

    """
    reactivity_columns = [colname for colname in cleared_train_data.columns if re.match(to_match, colname)]
    if len(reactivity_columns) == 0:
        raise Exception("Data frame does not contain columns to match with")

    # Target values
    targets = cleared_train_data[reactivity_columns].astype(dtype)

    return targets


def reactivity_masking(targets) :
    """Handle Na reactivity values by creating a mask

    Parameters:
        targets(DataFrame):
            dataframe with reactivity columns

    Returns:
        reactivity_mask(Boolean mask):
            mask for na reactivity values

    """
    reactivity_mask = ~np.isnan(targets.values)
    return reactivity_mask


def train_val_sets(*arrays, test_size=0.2, random_state=42):
    """Creates the Validation and the training sets

    Parameters :
        - features(matrix) : concatenated encoded matrix
        - targets(DataFrame) : dataframe with reactivity columns
        - reactivity_mask(Boolean mask) : mask for na reactivity values
    
    Returns :
        - X_train
        - X_val
        - y_train
        - y_val
        - mask_train
        - mask_val

    """
    return train_test_split(
                *arrays, test_size=test_size, random_state=random_state
            )

if __name__ == "__main__":
    X, Y = auxiliary.load_npy_xy("./data/X.npy", "./data/Y.npy")
    

