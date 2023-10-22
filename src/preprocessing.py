#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder
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


def filter_SN(data, sn_column="SN_filter"):
    """Filter the rows where "SN_filter" is equal to 1.

    Parameters:
        - data (DataFrame):
            A pandas Dataframe containing the input data

    Returns:
        - cleaned_train_data (DataFrame):
            A dataframe containing only the sequences that
            passed the SN_filter.
    """
    cleaned_train_data = data[data[sn_column] == 1]
    return cleaned_train_data


def filter_identical_sequences(
    data,
    group_column="sequence",
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
    filtered_df = data.groupby(group_column).apply(
        lambda x: x.loc[x[signal_column].idxmax()]
    )
    return filtered_df


def columns_defining(cleaned_train_data) :
    """Define the X and Y columns.

    Parameters:
        - cleaned_trained_data (Dataframe):
            pandas dataframe input

    Returns:
        - x_columns (list):
            column names for X
        - y_columns (list):
            column names for Y
        - conditional_columns (list):
            contionnal columns for the whole dataset
    """
    x_columns = ["sequence_id", "sequence"]
    conditional_columns = ["experiment_type", "signal_to_noise"]
    y_columns = [colname for colname in cleaned_train_data.columns if re.match("^reactivity_[0-9]{4}$", colname)]
    return x_columns, y_columns, conditional_columns


def column_filtering(cleaned_train_data, x_columns, y_columns, conditional_columns) :
    """Select the necessary columns in the dataframe

    Parameters:
        - cleaned_train_data(DataFrame) : dataframe input
        - x_columns(list): list of columns for X
        - y_columns(list): list of columns for Y
        - conditional_columns(list): list of conditional columns

    Returns:
        - cleaned_train_data(DataFrame) : dataframe with X and Y columns and the co,nditional ones
    """
    cleaned_train_data = cleaned_train_data[x_columns + conditional_columns + y_columns]
    return cleaned_train_data


def dataframe_separation(cleaned_train_data, experiment_column="experiment_type"):
    """Create two dataframe based on the experiment type (DMS and 2A3)

    Parameters:
        - cleaned_train_data(DataFrame):
            input dataframe to be separated in two

    Returns:
        - df_2A3_MaP(DataFrame):
            dataframe with the 2A3 results only
        - df_DMS_MaP(DataFrame):
            dataframe with the DMS results only
    """
    df_2A3_MaP = cleaned_train_data[cleaned_train_data[experiment_column] == '2A3_MaP']
    df_DMS_MaP = cleaned_train_data[cleaned_train_data[experiment_column] == 'DMS_MaP']

    return df_2A3_MaP, df_DMS_MaP 


def dataframe_concatenation(df_2A3_MaP, df_DMS_MaP):
    """Concatenation of the experiment dataframes

    Parameters :
        - df_2A3_MaP(DataFrame):
            dataframe with the 2A3 experiment results
        - df_DMS_MaP(DataFrame):
            dataframe with the DMS experiment results

    Returns :
        - cleared_train_data(DataFrame):
            concatenated dataframe
    """
    mask_2A3 = df_2A3_MaP["sequence"].isin(df_DMS_MaP["sequence"])
    mask_DMS = df_DMS_MaP["sequence"].isin(df_2A3_MaP["sequence"])

    cleared_train_data = pd.concat([df_2A3_MaP[mask_2A3], df_DMS_MaP[mask_DMS]], ignore_index=True)
    cleared_train_data = cleared_train_data.drop(columns=['signal_to_noise'], inplace=True)
    return cleared_train_data


def csv_export(cleared_train_data, y_columns) :
    """Export the dataframe into a csv file
    
    Parameters : 
    - cleared_train_data(DataFrame) : dataframe with the whole data
    - y_columns(list) : columns for Y data
    """
    #Convert the y columns     
    cleared_train_data[y_columns] = cleared_train_data[y_columns].astype(np.float32)
    
    #export the csv
    csv_path = input("Enter your path :") 
    cleared_train_data.to_csv(csv_path, index=False)


def extraction(data) :
    """Extract RNA sequences and experiment type
    
    Parameters :
    - data(Dataframe) : preloaded dataframe with the right format
    
    Returns:
    - rna_sequence(Series) : a pandas series with all the sequences
    - experiment_type(Series) : a panda series with the experiment types
    """
    rna_sequences = data['sequence']
    experiment_type = data['experiment_type']

    return rna_sequences, experiment_type


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
    """Returns the set of padded encoded sequences in one
    hot encoding for a list of sequences.

    """
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


def get_target(cleared_train_data, to_match="^reactivity_[0-9]{4}$", dtype=np.float32) :
    """Extract reactivity columns as targets to use as Y.

    Parameters:
        cleared_train_data(DataFrame):
            pandas dataframe with the whole data
    
    Returns:
        targets(DataFrame):
            dataframe with reactivity columns

    """
    
    reactivity_columns = cleared_train_data.columns[~cleared_train_data.columns.isin(['sequence','experiment_type'])]
    targets = cleared_train_data[reactivity_columns]
    
    return targets


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


def robust_z_normalization(y):
    y_median = np.nanmedian(y)
    y_mad = np.nanmedian(np.abs(np.array(y) - y_median))
    # Apply robust z-score normalization
    y_normalized = (y - y_median) / (1.482602218505602 * y_mad)

    return y_normalized


def get_y(data, y_col="2A3"):
    pass


if __name__ == "__main__":
    dt = auxiliary.load_npy("./data/ohe_cleared_train_data.npy", allow_pickle=True)
    X_list = []
    Y_list = []
    [
        (
            X_list.append(
                xy[['Nucleotide_A', 'Nucleotide_C', 'Nucleotide_G', 'Nucleotide_U']].tolist()
            ),
            Y_list.append(
                xy[['DMS_MaP_Reactivity', '2A3_MaP_Reactivity']].tolist()
            ),
        )
        for xy in dt
    ]

    import preprocessing
    X = preprocessing.pad_matrices(X_list)
    Y = preprocessing.pad_matrices(Y_list)

    x_train, x_val, x_test, y_train, y_val, y_test \
        = auxiliary.train_val_test_split(X, Y)

    auxiliary.save_npy(X, "./data/x",
                       Y, "./data/y",
                       x_train, "./data/x_train",
                       y_train, "./data/y_train",
                       x_val, "./data/x_val",
                       y_val, "./data/y_val",
                       x_test, "./data/x_test",
                       y_test, "./data/y_test")
