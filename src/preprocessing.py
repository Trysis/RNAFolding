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


def load_data(filepath, kwargs**) :
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

    data = pd.read_csv(filepath, kwargs**)
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


def features_encoding(rna_sequences, experiment_type):
    """Fit and transform RNA sequences.

    Parameters :
        - rna_sequence(Series):
            a pandas series with all the RNA sequences
        - experiment_type(Series):
            a panda series with the experiment types

    Returns :
        - features(matrix):
            concatenated encoded matrix
    """
    encoder = OneHotEncoder(sparse_output = True, dtype=np.int64)
    rna_sequences_encoded = encoder.fit_transform(rna_sequences.values.reshape(-1,1))
    experiment_type_encoded = pd.get_dummies(experiment_type)
    features = hstack((rna_sequences_encoded, experiment_type_encoded))
    return features


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


def train_val_sets(features, targets, reactivity_mask, test_size=0.2, random_state=42):
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
    
    X_train, X_val, y_train, y_val, mask_train, mask_val = train_test_split(
        features, targets, reactivity_mask, test_size=test_size, random_state=random_state)
    
    return X_train, X_val, y_train, y_val, mask_train, mask_val


def reshape_inp(X_train, X_val) :
    """Reshape the input format

    Parameters:
    - X_train: Training feature matrix (e.g., dense matrix)
    - X_val: Validation feature matrix (e.g., dense matrix)

    Returns:
    - X_train_reshaped: Reshaped training data with time step dimension
    - X_val_reshaped: Reshaped validation data with time step dimension
    """
    # Convert dense matrix to dense array 
    X_train_dense = X_train.toarray()
    X_val_dense = X_val.toarray()

    # Reshape input data to include the time step dimension
    timesteps = 1  # Number of time steps (since since we have masked sequences)
    input_dim = X_train_dense.shape[1]
    X_train_reshaped = X_train_dense.reshape(X_train_dense.shape[0], timesteps, input_dim)
    X_val_reshaped = X_val_dense.reshape(X_val_dense.shape[0], timesteps, input_dim)
    
    return X_val_reshaped, X_val_reshaped


if __name__ == "__main__":
    sequence = "AUCCUA"


