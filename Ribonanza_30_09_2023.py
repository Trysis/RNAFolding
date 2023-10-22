#!/usr/bin/env python
# coding: utf-8

# # **Load train data**

# Load train data

# In[3]:


a = 2
a



# In[7]:


import re
import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import OneHotEncoder


# In[8]:


# Constantes
RNA_BASES = [["A"], ["U"], ["C"], ["G"]]
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(RNA_BASES)


# In[9]:


train_data = pd.read_csv("/mnt/c/Users/lilol/Downloads/SN_filtered_train.csv", header=0)


# # **Exploratory Data Analysis (EDA)**

# To be completed: add analysis of shape of the data, pycharts, boxplots, pehaps PCA, univariate analysis, multivariate analysis...

# In[10]:


# Display the first 5 rows of the DataFrame
train_data.head()


# # **Feature engeneering**

# Remove "sequence_id", "dataset_name", "reads", "SN_filter" and "reactivity_error" columns

# In[11]:


# Function to keep rows with maximum signal_to_noise within identical sequences
def filter_identical_sequences(df):
    # Group by 'sequence' and keep the row with max 'signal_to_noise'
    filtered_df = df.groupby('sequence').apply(lambda x: x.loc[x['signal_to_noise'].idxmax()])
    return filtered_df


# In[12]:


def dict_from_data(data, keys_name=["2A3_MaP", "DMS_MaP"]):
    
    n_duplicate = sum(data.duplicated(subset=["sequence", "experiment_type"]))
    if n_duplicate > 0:
        return None

    seq_reactivity = dict()
    for seq, group in data.groupby("sequence"):
        seq_reactivity[seq] = dict()
        for key in keys_name:
            mask = group["experiment_type"] == key
            seq_reactivity[seq][key] = group[mask].drop(
                labels=["sequence", "experiment_type"], axis=1
            ).values.reshape(-1)
            seq_reactivity[seq][key] = np.expand_dims(seq_reactivity[seq][key], axis=1)

    return seq_reactivity

def XY_from_dict(dict_data, encoder, maxlen=457):
    x_list = []
    y_list = []
    for i, (sequence, reactivities) in enumerate(dict_data.items()):
        y = np.hstack([reactivities["2A3_MaP"], reactivities["DMS_MaP"]])
        x_list.append(onehot_from_sequence(sequence, encoder, maxlen=maxlen))
        y_list.append(padded_matrix(y, maxlen=maxlen))
    return np.array(x_list), np.array(y_list)

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
    if not isinstance(matrix_2d, np.ndarray):
        matrix_2d = np.array(matrix_2d)

    n_toadd = maxlen - matrix_2d.shape[0]
    padding = ((0, n_toadd), (0, 0))  # padding on axis
    matrix_2d_padded = np.pad(matrix_2d, pad_width=padding, mode="constant")
    return matrix_2d_padded


# In[13]:


# Get columns name for X, and Y
x_columns = ["sequence_id", "sequence"]
conditional_columns = ["experiment_type", "signal_to_noise"]
y_columns = [colname for colname in train_data.columns if re.match("^reactivity_[0-9]{4}$", colname)]

# Keep the necessary columns from the DataFrame
cleaned_train_data = train_data[x_columns + conditional_columns + y_columns]


# For each group of identical sequences, keep only the sequence with the highest signal to noise value

# In[14]:


# Create two separate DataFrames based on "experiment_type"
df_2A3_MaP = cleaned_train_data[cleaned_train_data['experiment_type'] == '2A3_MaP']
df_DMS_MaP = cleaned_train_data[cleaned_train_data['experiment_type'] == 'DMS_MaP']

# Delete cleaned_train_data to free space memory
del cleaned_train_data

df_2A3_MaP = filter_identical_sequences(df_2A3_MaP)  # Filter df_2A3_MaP
df_DMS_MaP = filter_identical_sequences(df_DMS_MaP)  # Filter df_DMS_MaP


# In[15]:


n_duplicate = sum(df_2A3_MaP.duplicated(subset=["sequence"]))


# In[16]:


print(n_duplicate)


# In[17]:


import pandas as pd
import numpy as np

def reactivity_normalization(df_2A3_MaP, df_DMS_MaP):
    """
    Robust Z-score Normalization of reactivities

    Parameters :

    - df_2A3_MaP(DataFrame) : dataframe with the 2A3 experiment results
    - df_DMS_MaP(DataFrame) : dataframe with the DMS experiment results

    Returns :

    - df_2A3_MaP(DataFrame) : Normalized dataframe with the 2A3 experiment results
    - df_DMS_MaP(DataFrame) : Normalized dataframe with the DMS experiment results
    """
    # Calculate median and Median Absolute Deviation (MAD) for robust normalization
    dms_median = np.nanmedian(df_2A3_MaP.iloc[:, 5:], axis=0)
    a3_median = np.nanmedian(df_DMS_MaP.iloc[:, 5:], axis=0)
    dms_mad = np.nanmedian(np.abs(df_DMS_MaP.iloc[:, 5:] - dms_median), axis=0)
    a3_mad = np.nanmedian(np.abs(df_2A3_MaP.iloc[:, 5:] - a3_median), axis=0)

    # Apply robust z-score normalization to all DataFrames from column 5 onwards
    df_DMS_MaP.iloc[:, 5:] = (df_DMS_MaP.iloc[:, 5:] - dms_median) / (1.482602218505602 * dms_mad)
    df_2A3_MaP.iloc[:, 5:] = (df_2A3_MaP.iloc[:, 5:] - a3_median) / (1.482602218505602 * a3_mad)

    return df_DMS_MaP, df_2A3_MaP


# In[18]:


reactivity_normalization(df_DMS_MaP, df_2A3_MaP)


# In[19]:


# Concatenate the two data frames
mask_2A3 = df_2A3_MaP["sequence"].isin(df_DMS_MaP["sequence"])
mask_DMS = df_DMS_MaP["sequence"].isin(df_2A3_MaP["sequence"])

cleared_train_data = pd.concat([df_2A3_MaP[mask_2A3], df_DMS_MaP[mask_DMS]], ignore_index=True)
cleared_train_data.drop(columns=['signal_to_noise'], inplace=True)


# Save the cleared train data into a csv file

# In[20]:


# columns type
cleared_train_data[y_columns] = cleared_train_data[y_columns].astype(np.float32)


# In[21]:


# Save cleared_train_data as a CSV file
csv_path = '~/RNAFolding/data/cleared_train_data.csv'
cleared_train_data.to_csv(csv_path, index=False)


# In[22]:


cleared_train_data.tail()


# In[ ]:


dict_data = dict_from_data(cleared_train_data)
x, y = XY_from_dict(dict_data, encoder)


# In[9]:


x.shape


# In[10]:


y.shape


# # **Model**
# (To be corrected)

# Load cleared train data

# In[350]:


# Define the path of the CSV file
csv_path = './data/cleared_train_data.csv'

# Load the CSV file as a DataFrame
cleared_train_data = pd.read_csv(csv_path)
dict_data = dict_from_data(cleared_train_data)
x, y = XY_from_dict(dict_data, encoder)


# Define features and targets

# In[390]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Reshape, Embedding
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack


# In[477]:


# Handle NaN values by creating a mask
x_mask = x.sum(axis=2) == 0
reactivity_mask = ~np.isnan(y)


# In[478]:


# Split the data into training and validation sets
x_train, x_val, y_train, y_val, mask_train, mask_val = train_test_split(
    x, y, x_mask, test_size=0.2, random_state=42
)


# RNN model

# In[479]:


y_train = np.nan_to_num(y_train)  # pour le moment on remplace les nan par des 0


# In[480]:


# Supposons que vous avez déjà créé votre masque de padding (padding_mask) comme indiqué précédemment.

# Créez votre modèle
model = keras.Sequential([
    keras.layers.Input(shape=(457, 4)),
    keras.layers.LSTM(units=8, return_sequences=True),
    keras.layers.Dense(units=2, activation='linear')
])

# Appliquez le masque de padding à la sortie de la couche LSTM
#masked_output = keras.layers.Masking(mask_value=True)(model.output)

# Créez un modèle final en utilisant la sortie masquée
final_model = keras.Model(inputs=model.input, outputs=model.output)

# Compilez et entraînez le modèle comme d'habitude
final_model.compile(loss='mean_squared_error', optimizer='adam')
final_model.fit(x_train, y_train, epochs=3, batch_size=16)


# In[483]:


model.predict(x)


# In[11]:


np.save("./data/X.data", x)
np.save("./data/Y.data", y)


# # **Load test**

# In[ ]:





# # **Check efficiency of the model**

# In[ ]:





# # **Save submission**

# In[ ]:





# # **Plot RNA structure**

# In[ ]:




