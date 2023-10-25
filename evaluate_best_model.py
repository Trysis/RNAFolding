import os
import matplotlib.pyplot as plt
from keras.models import load_model
from contextlib import redirect_stdout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
import pickle

# Initialize lists to store metrics
loaded_train_losses = []
loaded_val_losses = []
loaded_train_maes = []
loaded_val_maes = []
loaded_train_rmses = []
loaded_val_rmses = []

# Initialize a variable to track the last epoch
last_epoch = 1

# Load metrics from all available epochs
while True:
    metrics_filename = os.path.join(f"/shared/projects/master-bi/groupe_RNA/MERABET/training_metrics/metrics_epoch_{last_epoch}.txt")
    if os.path.exists(metrics_filename):
        with open(metrics_filename, 'r') as metrics_file:
            lines = metrics_file.read().split('\n')
            if len(lines) >= 6:
                loaded_train_losses.append(float(lines[0].split(': ')[1]))
                loaded_val_losses.append(float(lines[1].split(': ')[1]))
                loaded_train_maes.append(float(lines[2].split(': ')[1]))
                loaded_val_maes.append(float(lines[3].split(': ')[1]))
                loaded_train_rmses.append(float(lines[4].split(': ')[1]))
                loaded_val_rmses.append(float(lines[5].split(': ')[1]))
        last_epoch += 1
    else:
        break  # Stop when the file for the next epoch does not exist

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
model = load_model('/shared/projects/master-bi/groupe_RNA/MERABET/trained_model_1_val_loss_2.102303117749559_.h5', custom_objects=custom_objects)


# Load X_test from the file
#X_test = np.load('/shared/projects/master-bi/groupe_RNA/MERABET/X_test.npy', allow_pickle=True)
with open('/shared/projects/master-bi/groupe_RNA/MERABET/X_test.pkl', 'rb') as file:
    X_test = pickle.load(file)
# Load y_test from the file
#y_test = np.load('/shared/projects/master-bi/groupe_RNA/MERABET/y_test.npy', allow_pickle=True)
# Load the list of arrays from the file
with open('/shared/projects/master-bi/groupe_RNA/MERABET/y_test.pkl', 'rb') as file:
    y_test = pickle.load(file)

batch_size = 64

# Evaluate the model on the test data
test_steps = len(X_test) // batch_size
test_data_generator = data_generator(X_test, y_test, batch_size)

test_loss = 0.0
test_mae = 0.0
test_rmse = 0.0

with open('/dev/null', 'w') as null_file:
    with redirect_stdout(null_file):
        for _ in range(test_steps):
            X_batch, y_batch = next(test_data_generator)
            batch_loss = model.test_on_batch(X_batch, y_batch)

            # Calculate MAE and RMSE separately
            batch_mae = MeanAbsoluteError()(y_batch, model.predict(X_batch))
            batch_rmse = RootMeanSquaredError()(y_batch, model.predict(X_batch))

            # Apply the same masking as in the custom loss function
            nan_mask = tf.math.is_nan(y_batch)
            padded_mask = tf.math.equal(y_batch, 0.0)
            composite_mask = tf.math.logical_or(nan_mask, padded_mask)
            modified_y_batch = tf.where(composite_mask, 0.0, y_batch)

            test_loss += batch_loss
            test_mae += tf.reduce_mean(tf.abs(modified_y_batch - model.predict(X_batch)))
            test_rmse += tf.sqrt(tf.reduce_mean(tf.square(modified_y_batch - model.predict(X_batch))))

    test_loss /= test_steps
    test_mae /= test_steps
    test_rmse /= test_steps

print(f"Test Loss: {test_loss}, Test MAE: {test_mae}, Test RMSE: {test_rmse}")

# Define the best epoch
best_epoch = 7  # You can adjust this to the actual best epoch

# Plot Loss (MSE)
plt.figure(figsize=(10, 6))
plt.plot(range(1, last_epoch), loaded_train_losses, label='Training Loss')
plt.plot(range(1, last_epoch), loaded_val_losses, label='Validation Loss')
plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
plt.axvline(x=best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')  # Add a vertical line
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss (MSE) Over Epochs')
plt.legend()
plt.grid(True)

# Define the path to save the PDF file
save_path = '/shared/projects/master-bi/groupe_RNA/MERABET/loss_plot.pdf'

# Save the plot to a PDF file
plt.savefig(save_path, format='pdf', bbox_inches='tight')

# Show the plot (optional)
#plt.show()

# Plot RMSE
plt.figure(figsize=(10, 6))
plt.plot(range(1, last_epoch), loaded_train_rmses, label='Train RMSE')
plt.plot(range(1, last_epoch), loaded_val_rmses, label='Validation RMSE')
plt.axhline(y=test_rmse, color='r', linestyle='--', label='Test RMSE')
plt.axvline(x=best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.title('RMSE Over Epochs')
# Define the path to save the PDF file
save_path = '/shared/projects/master-bi/groupe_RNA/MERABET/RMSE_plot.pdf'

# Save the plot to a PDF file
plt.savefig(save_path, format='pdf', bbox_inches='tight')

# Show the plot (optional)
#plt.show()

# Plot MAE
plt.figure(figsize=(10, 6))
plt.plot(range(1, last_epoch), loaded_train_maes, label='Train MAE')
plt.plot(range(1, last_epoch), loaded_val_maes, label='Validation MAE')
plt.axhline(y=test_mae, color='r', linestyle='--', label='Test MAE')
plt.axvline(x=best_epoch, color='green', linestyle='--', label=f'Best Epoch ({best_epoch})')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.title('MAE Over Epochs')
# Define the path to save the PDF file
save_path = '/shared/projects/master-bi/groupe_RNA/MERABET/MAE_plot.pdf'

# Save the plot to a PDF file
plt.savefig(save_path, format='pdf', bbox_inches='tight')

# Show the plot (optional)
#plt.show()

# Convert y_test to a single NumPy array
#y_test_array = np.array([np.array(arr) for arr in y_test])
y_test_array = np.empty(len(y_test), dtype=object)
for i in range(len(y_test)):
    y_test_array[i] = np.array(y_test[i])

# Create an empty list to store the predictions
predictions = []

# Iterate through the sequences in X_test and make predictions
for input_sequence in X_test:
    # Predict reactivity values for the current RNA sequence using the loaded model
    predicted_value = model.predict(np.array(input_sequence).reshape(1, -1, 4))
    # Append the predicted value to the list of predictions
    predictions.append(predicted_value)
    
# Extract DMS values
y_test_dms = [y_test[:, 0] for y_test in y_test_array]

# To separate for all predictions, you can use a loop or list comprehension
y_test_pred_dms = [y_pred[0][:, 0] for y_pred in predictions]

# Flatten the nested arrays
flattened_y_test_dms = np.concatenate(y_test_dms)
flattened_y_test_pred_dms = np.concatenate(y_test_pred_dms)

# Find non-NaN indices in flattened_y_test_dms
non_nan_indices_dms = ~np.isnan(flattened_y_test_dms)

# Extract non-NaN values from flattened_y_test_dms and flattened_y_test_pred_dms
y_test_dms_non_nan = flattened_y_test_dms[non_nan_indices_dms]
y_test_pred_dms_non_nan = flattened_y_test_pred_dms[non_nan_indices_dms]

# Calculate the prediction errors
errors_dms = y_test_dms_non_nan - y_test_pred_dms_non_nan

# Create a histogram of prediction errors
plt.figure(figsize=(8, 6))
plt.hist(errors_dms, bins=50)
plt.xlabel('Prediction Errors')
plt.ylabel('Frequency')
plt.title('Histogram of Prediction Errors - DMS experiment')
# Define the path to save the PDF file
save_path = '/shared/projects/master-bi/groupe_RNA/MERABET/Histogram_DMS.pdf'

# Save the plot to a PDF file
plt.savefig(save_path, format='pdf', bbox_inches='tight')

# Show the plot (optional)
#plt.show()

# Calculate prediction errors
y_test_2a3 = [y_test[:, 1] for y_test in y_test_array]

# To separate for all predictions, you can use a loop or list comprehension
y_test_pred_2a3 = [y_pred[0][:, 1] for y_pred in predictions]

# Flatten the nested arrays
flattened_y_test_2a3 = np.concatenate(y_test_2a3)
flattened_y_test_pred_2a3 = np.concatenate(y_test_pred_2a3)

# Find non-NaN indices in flattened_y_test_2a3
non_nan_indices_2a3 = ~np.isnan(flattened_y_test_2a3)

# Extract non-NaN values from flattened_y_test_2a3 and flattened_y_test_pred_2a3
y_test_2a3_non_nan = flattened_y_test_2a3[non_nan_indices_2a3]
y_test_pred_2a3_non_nan = flattened_y_test_pred_2a3[non_nan_indices_2a3]

# Calculate the prediction errors
errors_2a3 = y_test_2a3_non_nan - y_test_pred_2a3_non_nan

# Create a histogram of prediction errors
plt.figure(figsize=(8, 6))
plt.hist(errors_2a3, bins=50)
plt.xlabel('Prediction Errors')
plt.ylabel('Frequency')
plt.title('Histogram of Prediction Errors - 2A3 experiment')
# Define the path to save the PDF file
save_path = '/shared/projects/master-bi/groupe_RNA/MERABET/Histogram_2A3.pdf'

# Save the plot to a PDF file
plt.savefig(save_path, format='pdf', bbox_inches='tight')

# Show the plot (optional)
#plt.show()

# Create a scatter plot for non-NaN predicted vs. actual values
plt.figure(figsize=(8, 8))
plt.scatter(y_test_dms_non_nan, y_test_pred_dms_non_nan)
plt.xlabel('Actual Values - DMS')
plt.ylabel('Predicted Values DMS')
plt.title('Scatter Plot of Predicted vs. Actual Values - DMS')

# Define the path to save the PDF file
save_path = '/shared/projects/master-bi/groupe_RNA/MERABET/Scatter_Plot_DMS.pdf'

# Save the plot to a PDF file
plt.savefig(save_path, format='pdf', bbox_inches='tight')

# Show the plot (optional)
#plt.show()

# Create a scatter plot for non-NaN predicted vs. actual values
plt.figure(figsize=(8, 8))
plt.scatter(y_test_2a3_non_nan, y_test_pred_2a3_non_nan)
plt.xlabel('Actual Values 2A3')
plt.ylabel('Predicted Values 2A3')
plt.title('Scatter Plot of Predicted vs. Actual Values - 2A3 experiment')

# Define the path to save the PDF file
save_path = '/shared/projects/master-bi/groupe_RNA/MERABET/Scatter_Plot_2A3.pdf'

# Save the plot to a PDF file
plt.savefig(save_path, format='pdf', bbox_inches='tight')

# Show the plot (optional)
#plt.show()

# Create a scatter plot for non-NaN predicted vs. actual values
plt.figure(figsize=(8, 8))
plt.scatter(y_test_dms_non_nan, y_test_pred_dms_non_nan)
plt.xlabel('Actual Values DMS')
plt.ylabel('Predicted Values DMS')
plt.title('Scatter Plot of Predicted vs. Actual Values - DMS experiment')

# Force equal aspect ratio on both axes
plt.axis('equal')

# Define the path to save the PDF file
save_path = '/shared/projects/master-bi/groupe_RNA/MERABET/Scatter_Plot_DMS_with_equal.pdf'

# Save the plot to a PDF file
plt.savefig(save_path, format='pdf', bbox_inches='tight')

# Show the plot (optional)
#plt.show()

# Create a scatter plot for non-NaN predicted vs. actual values
plt.figure(figsize=(8, 8))
plt.scatter(y_test_2a3_non_nan, y_test_pred_2a3_non_nan)
plt.xlabel('Actual Values 2A3')
plt.ylabel('Predicted Values 2A3')
plt.title('Scatter Plot of Predicted vs. Actual Values - 2A3 experiment')

# Force equal aspect ratio on both axes
plt.axis('equal')

# Define the path to save the PDF file
save_path = '/shared/projects/master-bi/groupe_RNA/MERABET/Scatter_Plot_2A3_with_equal.pdf'

# Save the plot to a PDF file
plt.savefig(save_path, format='pdf', bbox_inches='tight')

# Show the plot (optional)
#plt.show()

# Distribution of DMS test reactivity values
plt.hist(y_test_dms_non_nan, bins=20, edgecolor='k')
plt.title("Distribution of DMS test reactivity values")
plt.xlabel("Values")
plt.ylabel("Frequency")
# Define the path to save the PDF file
save_path = '/shared/projects/master-bi/groupe_RNA/MERABET/DMS_test_reactivity_values.pdf'

# Save the plot to a PDF file
plt.savefig(save_path, format='pdf', bbox_inches='tight')

# Show the plot (optional)
#plt.show()

# Distribution of predicted DMS test reactivity values
plt.hist(y_test_pred_dms_non_nan, bins=20, edgecolor='k')
plt.title("Distribution of predicted DMS test reactivity values")
plt.xlabel("Values")
plt.ylabel("Frequency")
# Define the path to save the PDF file
save_path = '/shared/projects/master-bi/groupe_RNA/MERABET/predicted_DMS_test_reactivity_values.pdf'

# Save the plot to a PDF file
plt.savefig(save_path, format='pdf', bbox_inches='tight')

# Show the plot (optional)
#plt.show()

# Distribution of predicted DMS test reactivity values
plt.hist(y_test_2a3_non_nan, bins=20, edgecolor='k')
plt.title("Distribution of 2A3 test reactivity values")
plt.xlabel("Values")
plt.ylabel("Frequency")
# Define the path to save the PDF file
save_path = '/shared/projects/master-bi/groupe_RNA/MERABET/2A3_test_reactivity_values.pdf'

# Save the plot to a PDF file
plt.savefig(save_path, format='pdf', bbox_inches='tight')

# Show the plot (optional)
#plt.show()

# Distribution of predicted DMS test reactivity values
plt.hist(y_test_pred_2a3_non_nan, bins=20, edgecolor='k')
plt.title("Distribution of predicted 2A3 test reactivity values")
plt.xlabel("Values")
plt.ylabel("Frequency")
# Define the path to save the PDF file
save_path = '/shared/projects/master-bi/groupe_RNA/MERABET/predicted_2A3_test_reactivity_values.pdf'

# Save the plot to a PDF file
plt.savefig(save_path, format='pdf', bbox_inches='tight')

# Show the plot (optional)
#plt.show()