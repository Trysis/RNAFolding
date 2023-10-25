import pandas as pd
import numpy as np
from scipy import stats
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
import sys
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from math import inf
import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import tensorflow as tf
from keras.models import load_model
import pickle

# Define the path of the CSV file
npy_path = '/shared/projects/master-bi/groupe_RNA/ohe_cleared_train_data.npy'

# Load the NPY file
ohe_cleared_train_data = np.load(npy_path, allow_pickle=True)

#ohe_cleared_train_data = ohe_cleared_train_data[0:1000]

# Define custom loss which is a custom object
def custom_loss(y_true, y_pred, padded_value=0.0):
    nan_mask = tf.math.is_nan(y_true)
    padded_mask = tf.math.equal(y_true, padded_value)

    # Create a composite mask that identifies both NaN and padded values
    composite_mask = tf.math.logical_or(nan_mask, padded_mask)

    # Replace NaN and padded values with 0.0
    modified_true = tf.where(composite_mask, 0.0, y_true)

    # Calculate the mean squared difference (MSE)
    loss = tf.reduce_mean(tf.square(modified_true - y_pred))

    return loss

# Define a dictionary of custom objects to be used during model loading
custom_objects = {'custom_loss': custom_loss}
# Load the model
model = load_model('/shared/projects/master-bi/groupe_RNA/MERABET/model_trial_20.h5', custom_objects=custom_objects)

# Define batch size
batch_size = 64

# Shuffle the data
random.shuffle(ohe_cleared_train_data)

# Split ratios
train_ratio = 0.6  # 60% for training
val_ratio = 0.2    # 20% for validation
test_ratio = 0.2   # 20% for testing

# Calculate the split indices
total_samples = len(ohe_cleared_train_data)
train_split_index = int(total_samples * train_ratio)
val_split_index = int(total_samples * (train_ratio + val_ratio))

# Split the data into train_data, val_data, and test_data
X_train = [seq[['Nucleotide_A', 'Nucleotide_C', 'Nucleotide_G', 'Nucleotide_U']].tolist() for seq in ohe_cleared_train_data[:train_split_index]]
y_train = [seq[['DMS_MaP_Reactivity', '2A3_MaP_Reactivity']].tolist() for seq in ohe_cleared_train_data[:train_split_index]]
X_val = [seq[['Nucleotide_A', 'Nucleotide_C', 'Nucleotide_G', 'Nucleotide_U']].tolist() for seq in ohe_cleared_train_data[train_split_index:val_split_index]]
y_val = [seq[['DMS_MaP_Reactivity', '2A3_MaP_Reactivity']].tolist() for seq in ohe_cleared_train_data[train_split_index:val_split_index]]
X_test = [seq[['Nucleotide_A', 'Nucleotide_C', 'Nucleotide_G', 'Nucleotide_U']].tolist() for seq in ohe_cleared_train_data[val_split_index:]]
y_test = [seq[['DMS_MaP_Reactivity', '2A3_MaP_Reactivity']].tolist() for seq in ohe_cleared_train_data[val_split_index:]]

# Save X_test to a file
#np.save('/shared/projects/master-bi/groupe_RNA/MERABET/X_test.npy', X_test)
with open('/shared/projects/master-bi/groupe_RNA/MERABET/X_test.pkl', 'wb') as file:
    pickle.dump(X_test, file)
# Save y_test to a file
#np.save('/shared/projects/master-bi/groupe_RNA/MERABET/y_test.npy', y_test)
with open('/shared/projects/master-bi/groupe_RNA/MERABET/y_test.pkl', 'wb') as file:
    pickle.dump(y_test, file)

# Create a generator function to yield batches of data with RNA sequence padding
def data_generator(X, y, batch_size):
    num_samples = len(X)
    while True:
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            X_batch = X[start:end]
            y_batch = y[start:end]

            # Pad RNA sequences to the length of the longest sequence within the batch
            max_length = max(len(seq) for seq in X_batch)
            X_batch = pad_sequences(X_batch, padding='post', dtype='float32', maxlen=max_length)
            y_batch = pad_sequences(y_batch, padding='post', dtype='float32', maxlen=max_length)

            yield X_batch, y_batch

def custom_loss(y_true, y_pred, padded_value=0.0):
    nan_mask = tf.math.is_nan(y_true)
    padded_mask = tf.math.equal(y_true, padded_value)

    # Create a composite mask that identifies both NaN and padded values
    composite_mask = tf.math.logical_or(nan_mask, padded_mask)

    # Replace NaN and padded values with 0.0
    modified_true = tf.where(composite_mask, 0.0, y_true)

    # Calculate the mean squared difference (MSE)
    loss = tf.reduce_mean(tf.square(modified_true - y_pred))

    return loss

# Fit the model using the generator
train_steps = len(X_train) // batch_size
val_steps = len(X_val) // batch_size

# Create lists to store loss, MAE, and RMSE values for training and validation
train_losses = []
val_losses = []
train_maes = []
val_maes = []
train_rmses = []
val_rmses = []

train_data_generator = data_generator(X_train, y_train, batch_size)
val_data_generator = data_generator(X_val, y_val, batch_size)

num_epochs = 10000
# Define early stopping parameters
min_delta = 0  # Minimum change in validation loss to be considered an improvement
patience = 10   # Number of epochs with no improvement before stopping
best_val_loss = float('inf')  # Initialize the best validation loss
wait = 0  # Counter for patience

# Model's epoch number
model_epoch = 0

# Initialize the best validation loss and best epoch
best_val_loss = float('inf')
best_epoch = 0

for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    train_mae = 0.0
    val_mae = 0.0
    train_rmse = 0.0
    val_rmse = 0.0

    # Redirect stdout to /dev/null
    with open('/dev/null', 'w') as null_file:
        with redirect_stdout(null_file):
            for _ in range(train_steps):
                X_batch, y_batch = next(train_data_generator)
                batch_loss = model.train_on_batch(X_batch, y_batch)

                # Calculate MAE and RMSE separately
                batch_mae = MeanAbsoluteError()(y_batch, model.predict(X_batch))
                batch_rmse = RootMeanSquaredError()(y_batch, model.predict(X_batch))

                # Apply the same masking as in the custom loss function
                nan_mask = tf.math.is_nan(y_batch)
                padded_mask = tf.math.equal(y_batch, 0.0)
                composite_mask = tf.math.logical_or(nan_mask, padded_mask)
                modified_y_batch = tf.where(composite_mask, 0.0, y_batch)

                train_loss += batch_loss
                train_mae += tf.reduce_mean(tf.abs(modified_y_batch - model.predict(X_batch)))
                train_rmse += tf.sqrt(tf.reduce_mean(tf.square(modified_y_batch - model.predict(X_batch))))

            for _ in range(val_steps):
                X_batch, y_batch = next(val_data_generator)
                batch_loss = model.test_on_batch(X_batch, y_batch)

                # Calculate MAE and RMSE separately
                batch_mae = MeanAbsoluteError()(y_batch, model.predict(X_batch))
                batch_rmse = RootMeanSquaredError()(y_batch, model.predict(X_batch))

                # Apply the same masking as in the custom loss function
                nan_mask = tf.math.is_nan(y_batch)
                padded_mask = tf.math.equal(y_batch, 0.0)
                composite_mask = tf.math.logical_or(nan_mask, padded_mask)
                modified_y_batch = tf.where(composite_mask, 0.0, y_batch)

                val_loss += batch_loss
                val_mae += tf.reduce_mean(tf.abs(modified_y_batch - model.predict(X_batch)))
                val_rmse += tf.sqrt(tf.reduce_mean(tf.square(modified_y_batch - model.predict(X_batch))))

    train_losses.append(train_loss / train_steps)
    val_losses.append(val_loss / val_steps)
    
    train_maes.append(train_mae / train_steps)
    val_maes.append(val_mae / val_steps)
    train_rmses.append(train_rmse / train_steps)
    val_rmses.append(val_rmse / val_steps)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1  # Note that epochs are 0-indexed

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / train_steps}, Validation Loss: {val_loss / val_steps}, Train MAE: {train_mae / train_steps}, Validation MAE: {val_mae / val_steps}, Train RMSE: {train_rmse / train_steps}, Validation RMSE: {val_rmse / val_steps}, Best Epoch: {best_epoch}")
    
    # Save the model at the end of each epoch
    model_epoch += 1
    model.save(f'/shared/projects/master-bi/groupe_RNA/MERABET/trained_model_{model_epoch}_val_loss_{val_loss / val_steps}_.h5')

    # Save metrics to a file
    metrics_filename = f"/shared/projects/master-bi/groupe_RNA/MERABET/training_metrics/metrics_epoch_{epoch + 1}.txt"
    with open(metrics_filename, 'w') as metrics_file:
        metrics_file.write(f"Train Loss: {train_loss / train_steps}\n")
        metrics_file.write(f"Validation Loss: {val_loss / val_steps}\n")
        metrics_file.write(f"Train MAE: {train_mae / train_steps}\n")
        metrics_file.write(f"Validation MAE: {val_mae / val_steps}\n")
        metrics_file.write(f"Train RMSE: {train_rmse / train_steps}\n")
        metrics_file.write(f"Validation RMSE: {val_rmse / val_steps}\n")

    # Check for early stopping
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        wait = 0  # Reset the patience counter
    else:
        wait += 1  # Increment the patience counter

    if wait >= patience:
        print("Early stopping triggered.")
        break