#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Reshape
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

# Constantes
RNA_BASES = [["A"], ["U"], ["C"], ["G"]]
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(RNA_BASES)


def load_data(file_path) :
    """Load data from a CSV file.

    Parameters:
    - file_path (str): The path to the CSV file containing the data.

    Returns:
    - data(DataFrame): A Pandas DataFrame containing the loaded data.
    """
    data = pd.read.csv(file_path)
    return data


def filter_SN(data):
    """Filter the rows where "SN_filter" is equal to 1 
       
    Parameters :
    - data (DataFrame) : A pandas Dataframe containing the input data
    
    Returns :
    - cleaned_train_data(DataFrame) : A dataframe containing only the sequences that passed the SN_filter 
    """
    cleaned_train_data = data[data['SN_filter'] == 1]
    return cleaned_train_data


def filter_identical_sequences(data):
    """For identical sequences keep rows with maximum signal to noise

    Parameters:
    - data (DataFrame) : pandas input dataframe

    Returns :
    - filtered_df(Dataframe) : a filtered dataframe with the sequences with a maximum signal to noise

    """
    filtered_df = cleaned_train_data.groupby('sequence').apply(lambda x: x.loc[x['signal_to_noise'].idxmax()])
    return filtered_df


def columns_defining(cleaned_train_data) :
    """Define the X and Y columns
    
    Parameters:
    -cleaned_trained_data (Dataframe) : pandas dataframe input
    
    Returns :
    -x_columns(list) : column names for X
    -y_columns(list) : column names for Y
    -conditional_columns(list) : continonal columns for the whole dataset
    """
    x_columns = ["sequence_id", "sequence"]
    conditional_columns = ["experiment_type", "signal_to_noise"]
    y_columns = [colname for colname in cleared_train_data.columns if re.match("^reactivity_[0-9]{4}$", colname)]
    return x_columns, y_columns, conditional_columns


def column_filtering(cleaned_train_data, x_columns, y_columns, conditional_columns) :
    
    """
    Select the necessary columns in the dataframe
    
    Parameters :
    -cleaned_train_data(DataFrame) : dataframe input
    -x_columns(list): list of columns for X
    -y_columns(list): list of columns for Y
    -conditional_columns(list): list of conditional columns
    
    Returns : 
    -cleaned_train_data(DataFrame) : dataframe with X and Y columns and the co,nditional ones
    """
    
    cleaned_train_data = cleaned_train_data[x_columns + conditional_columns + y_columns]
    return cleaned_train_data


# In[19]:


def dataframe_separation(cleaned_train_data):
    
    """
    Create two dataframe based on the experiment type (DMS and 2A3)
    
    Parameters :
    -cleaned_train_data(DataFrame) : input dataframe to be separated in two
    
    Returns :
    -df_2A3_MaP(DataFrame) : dataframe with the 2A3 results only
    -df_DMS_MaP(DataFrame) : dataframe with the DMS results only
    
    """
    df_2A3_MaP = cleaned_train_data[cleaned_train_data['experiment_type'] == '2A3_MaP']
    df_DMS_MaP = cleaned_train_data[cleaned_train_data['experiment_type'] == 'DMS_MaP']
    
    # Delete cleaned_train_data to free space memory
    del cleaned_train_data
    
    return df_2A3_MaP, df_DMS_MaP 


# In[20]:


def dataframe_concatenation(df_2A3_MaP, df_DMS_MaP) :
    
    """
    Concatenation of the experiment dataframes
    
    Parameters :
    - df_2A3_MaP(DataFrame) : dataframe with the 2A3 experiment results
    - df_DMS_MaP(DataFrame) : dataframe with the DMS experiment results
    
    Returns :
    
    -cleared_train_data(DataFrame) : concatenated dataframe
    """
    
    mask_2A3 = df_2A3_MaP["sequence"].isin(df_DMS_MaP["sequence"])
    mask_DMS = df_DMS_MaP["sequence"].isin(df_2A3_MaP["sequence"])
    
    cleared_train_data = pd.concat([df_2A3_MaP[mask_2A3], df_DMS_MaP[mask_DMS]], ignore_index=True)
    cleared_train_data = cleared_train_data.drop(columns=['signal_to_noise'], inplace=True)
    
    return cleared_train_data


# In[21]:


def csv_export(cleared_train_data, y_columns) :
    
    """
    Export the dataframe into a csv file
    
    Parameters : 
    - cleared_train_data(DataFrame) : dataframe with the whole data
    - y_columns(list) : columns for Y data
    """
    
    #Convert the y columns     
    cleared_train_data[y_columns] = cleared_train_data[y_columns].astype(np.float32)
    
    #export the csv
    csv_path = input("Enter your path :") 
    cleared_train_data.to_csv(csv_path, index=False)
    


# In[22]:


def extraction(data) :
    
    """
    Extract RNA sequences and experiment type
    
    Parameters :
    - data(Dataframe) : preloaded dataframe with the right format
    
    Returns:
    - rna_sequence(Series) : a pandas series with all the sequences
    - experiment_type(Series) : a panda series with the experiment types
    """
    
    rna_sequences = data['sequence']
    experiment_type = data['experiment_type']
    
    return rna_sequence, experiment_type


# In[23]:


def features_encoding(rna_sequences, experiment_type) :
    
    """
    Fit and transform RNA sequences
    
    Parameters :
    - rna_sequence(Series) : a pandas series with all the RNA sequences
    - experiment_type(Series) : a panda series with the experiment types
    
    Returns :
    - features(matrix) : concatenated encoded matrix
    
    """
    
    encoder = OneHotEncoder(sparse_output = True, dtype=np.int64)
    rna_sequences_encoded = encoder.fit_transform(rna_sequences.values.reshape(-1,1))
    experiment_type_encoded = pd.get_dummies(experiment_type)
    features = hstack((rna_sequences_encoded, experiment_type_encoded))
    return features


# In[24]:


def get_target(cleared_train_data) :
    
    """
    Extract reactivity columns as targets to use as Y
    
    Parameters :
    cleared_train_data(DataFrame) : pandas dataframe with the whole data
    
    Returns : 
    targets(DataFrame) : dataframe with reactivity columns
    """
    
    reactivity_columns = cleared_train_data.columns[~cleared_train_data.columns.isin(['sequence','experiment_type'])]
    targets = cleared_train_data[reactivity_columns]
    
    return targets


# In[ ]:


def reactivity_masking(targets) :
    
    """
    Handle Na reactivity values by creating a mask
    
    Parameters : 
    targets(DataFrame) : dataframe with reactivity columns
    
    Returns :
    reactivity_mask(Boolean mask) : mask for na reactivity values
    """
    
    reactivity_mask = ~np.isnan(targets.values)
    return reactivity_mask


# In[ ]:


def train_val_sets(features, targets, reactivity_mask):
    
    """
    Creates the Validation and the training sets
    
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
        features, targets, reactivity_mask, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val, mask_train, mask_val


# In[ ]:


def reshape_inp(X_train, X_val) :
    """
    Reshape the input format

    Parameters:
    - X_train: Training feature matrix (e.g., dense matrix)
    - X_val: Validation feature matrix (e.g., dense matrix)

    Returns:
    - X_train_reshaped: Reshaped training data with time step dimension
    - X_val_reshaped: Reshaped validation data with time step dimension
    """
    
    #Convert dense matrix to dense array 
    
    X_train_dense = X_train.toarray()
    X_val_dense = X_val.toarray()
    
    # Reshape input data to include the time step dimension
    timesteps = 1  # Number of time steps (since since we have masked sequences)
    input_dim = X_train_dense.shape[1]
    X_train_reshaped = X_train_dense.reshape(X_train_dense.shape[0], timesteps, input_dim)
    X_val_reshaped = X_val_dense.reshape(X_val_dense.shape[0], timesteps, input_dim)
    
    return X_val_reshaped, X_val_reshaped


# In[ ]:




