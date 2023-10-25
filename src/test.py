"""This script will be used to test the model on an X data."""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import keras

# Local modules
import model
import plots
import auxiliary


def test_model(model, x, y, id=None, 
               worst_r2=0.2, best_r2=0.7,
               model_name="unknown", metric="",
               save_to=None, 
               lab=None,
               overwrite=False
):
    """"""
    if save_to is not None and auxiliary.isdir(save_to):
        save_to = auxiliary.to_dirpath(save_to)
    else:
        raise Exception("Output directory is incorrect")

    y_obs, y_pred = y, model.predict(x)
    xlabel, ylabel = "valeur observée", "valeur prédite"
    save_to = auxiliary.create_dir(save_to + model_name, add_suffix=overwrite)

    to_title = f"\n{model_name}" if lab is None else f"\n{model_name}\n{lab}"
    # When ID is set
    if id is not None:
        y_obs_global = y_obs.reshape(-1, 2)
        y_pred_global = y_pred.reshape(-1, 2)

        # 2A3
        ## Global plot with 2A3 (all predicted - observed values)
        indices_global = np.arange(y_obs_global.shape[0])
        isnotnan = ~np.isnan(y_obs_global[:, 0]) & ~np.isnan(y_pred_global[:, 0])
        r2_2A3_global = r2_score(y_obs_global[:, 0][isnotnan], y_pred_global[:, 0][isnotnan])
        title_2A3_global = "2A3"
        for mode in ["scatter", "hist"]:
            filename_2A3_global = f"{mode}_R2={r2_2A3_global:2.4f}_{title_2A3_global}"
            filename_2A3_global = f"{lab}_{filename_2A3_global}" if lab is not None \
                                    else filename_2A3_global

            plots.plot(indices_global,
                       y_obs_global[:, 0], y_pred_global[:, 0], mode=mode,
                       title=f"{title_2A3_global}{to_title}",
                       metric=metric, r2=r2_2A3_global,
                       xlabel=xlabel, ylabel=ylabel,
                       filename=filename_2A3_global,
                       alphas=(0.4, 0.4),
                       forcename=True,
                       save_to=save_to
                      )

            plt.clf()

        # DMS
        ## Global plot with DMS (all predicted - observed values)
        isnotnan = ~np.isnan(y_obs_global[:, 1]) & ~np.isnan(y_pred_global[:, 1])
        r2_DMS_global = r2_score(y_obs_global[:, 1][isnotnan], y_pred_global[:, 1][isnotnan])
        title_DMS_global = "DMS"
        for mode in []:
            filename_DMS_global = f"{mode}_R2={r2_DMS_global:2.4f}_{title_DMS_global}"
            filename_DMS_global = f"{lab}_{filename_DMS_global}" if lab is not None \
                                else filename_DMS_global

            plots.plot(indices_global,
                    y_obs_global[:, 1], y_pred_global[:, 1], mode=mode,
                    title=f"{title_DMS_global}{to_title}",
                    metric=metric, r2=r2_DMS_global,
                    xlabel=xlabel, ylabel=ylabel,
                    filename=filename_DMS_global,
                    alphas=(0.4, 0.4),
                    forcename=True,
                    save_to=save_to
                    )

            plt.clf()

        # 2A3 & DMS
        # Plot by observations (each sequence)
        for id_seq, x_val, r_obs, r_pred in zip(id, x, y_obs, y_pred):
            title_2A3 = f"{id_seq} - 2A3" + to_title
            title_DMS = f"{id_seq} - DMS" + to_title
            # indices
            non_zero_idx = np.nonzero(np.sum(x_val, axis=1))
            idx_nan_end = np.where(non_zero_idx)[0][-1]
            indices = np.arange(idx_nan_end)

            # Plot 2A3
            ## Metrics
            isnotnan = ~np.isnan(r_obs[:, 1]) & ~np.isnan(r_pred[:, 1])
            r2_2A3 = r2_score(r_obs[:, 1][isnotnan], r_pred[:, 1][isnotnan])
            filename_2A3 = f"R2={r2_2A3:2.4f}_len={idx_nan_end}_{id_seq}_2A3"
            filename_2A3 = f"{lab}_{filename_2A3}" if lab is not None else filename_2A3
            plots.plot(indices, r_obs[:, 1], r_pred[:, 1],
                       title=title_2A3, metric=metric, r2=r2_2A3,
                       xlabel=xlabel, ylabel=ylabel,
                       filename=filename_2A3, alphas=(0.7, 0.4),
                       forcename=True,
                       save_to=save_to)

            plt.clf()  # Clear plot

            # Plot DMS
            ## Metrics
            isnotnan = ~np.isnan(r_obs[:, 1]) & ~np.isnan(r_pred[:, 1])
            r2_DMS = r2_score(r_obs[:, 1][isnotnan], r_pred[:, 1][isnotnan])
            filename_DMS = f"R2={r2_DMS:2.4f}_{id_seq}_DMS"
            filename_DMS = f"{lab}_{filename_DMS}" if lab is not None else filename_DMS
            plots.plot(indices, r_obs[:, 1], r_pred[:, 1],
                       title=title_DMS, metric=metric, r2=r2_DMS,
                       xlabel=xlabel, ylabel=ylabel,
                       filename=filename_DMS, alphas=(0.7, 0.4),
                       forcename=True,
                       save_to=save_to)

            plt.close("all")

    # When ID is not set
    else:
        for r_obs, r_pred in zip(y_obs, y_pred):
            to_title = "" if lab is None else f"\n{lab}"
            title_2A3 = f"2A3" + to_title
            title_DMS = f"DMS" + to_title
            indices = np.arange(r_obs.shape[0])

            # Plot 2A3
            ## Metrics
            isnotnan = ~np.isnan(r_obs[:, 0]) & ~np.isnan(r_pred[:, 0])
            r2_2A3 = r2_score(r_obs[:, 0][isnotnan], r_pred[:, 0][isnotnan])
            filename_2A3 = f"R2={r2_2A3:.4f}_2A3"
            filename_2A3 = f"{lab}_{filename_2A3}" if lab is not None else filename_2A3
            plots.plot(indices, r_obs[:, 0], r_pred[:, 0],
                       title=title_2A3, metric=metric, r2=r2_2A3,
                       xlabel=xlabel, ylabel=ylabel,
                       filename=filename_2A3,
                       forcename=True,
                       save_to=save_to)

            plt.clf()  # Clear plot

            # Plot DMS
            ## Metrics
            isnotnan = ~np.isnan(r_obs[:, 1]) & ~np.isnan(r_pred[:, 1])
            r2_DMS = r2_score(r_obs[:, 1][isnotnan], r_pred[:, 1][isnotnan])
            filename_DMS = f"R2={r2_DMS:.4f}_DMS"
            filename_DMS = f"{lab}_{filename_DMS}" if lab is not None else filename_DMS
            plots.plot(indices, r_obs[:, 1], r_pred[:, 1],
                       title=title_DMS, metric=metric, r2=r2_DMS,
                       xlabel=xlabel, ylabel=ylabel,
                       filename=filename_DMS,
                       forcename=True,
                       save_to=save_to)

            plt.close()


def read_test_submission(kaggle_format, save_to):
    """Does not work for the moment"""
    # Create a scatter Plot of Predicted vs. Actual Values
    # Create lists to store non-NaN values and corresponding predictions
    non_nan_y_values = []
    non_nan_predictions = []

    # Iterate through the elements in y_test and predictions
    for y_arr, prediction_arr in zip(y_test_array, model.predict(X_test)):
        # Filter out NaN values from the arrays
        non_nan_y = y_arr[~np.isnan(y_arr)]
        non_nan_prediction = prediction_arr[~np.isnan(y_arr)]  # Exclude corresponding NaN values

        # Append non-NaN values to the lists
        non_nan_y_values.append(non_nan_y)
        non_nan_predictions.append(non_nan_prediction)

    # Convert the lists to NumPy arrays for plotting
    y_test_non_nan = np.array(non_nan_y_values)
    predictions_non_nan = np.array(non_nan_predictions)

    # Flatten the non-NaN values in y_test_non_nan and predictions_non_nan
    y_test_non_nan = np.concatenate(y_test_non_nan)
    predictions_non_nan = np.concatenate(predictions_non_nan)

    # Create a scatter plot for non-NaN predicted vs. actual values
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_non_nan, predictions_non_nan)
    plt.xlabel('Actual Values (Column 1)')
    plt.ylabel('Predicted Values (Column 1)')
    plt.title('Scatter Plot of Predicted vs. Actual Values')
    plt.show()
    if kaggle_format:
        # Create the "sample_submission" DataFrame
        sample_submission = pd.DataFrame({
            'id': id.values,
            'reactivity_DMS_MaP': y.values[:, 1],
            'reactivity_2A3_MaP': y.values[:, 0]
        })

        # Save the sample_submission DataFrame to a CSV file
        sample_path = save_to + 'sample_submission.csv'
        sample_submission.to_csv(sample_path, index=False)
        print(f"Save sample submission to {sample_path}")


if __name__ == "__main__":
    ## Parsing
    parser = argparse.ArgumentParser(
            prog = "Test.py",
            description = "This program test a model on a chosen data (X).",
            epilog = ""
            )

    # Input Data
    parser.add_argument('model', type=str, help="File path to saved model file.")
    parser.add_argument('x_path', nargs='?', type=str, default="./data/x_val.npy", help="Filepath to X file.")
    parser.add_argument('y_path', nargs='?', type=str, default="./data/y_val.npy", help="Filepath to Y file.")
    parser.add_argument('-i', '--identifiant', type=str, default="./data/i_val.npy", help="Filepath indicating id associated with X")
    parser.add_argument('-o', '--output_dir', type=str, default="./out/", help="output directory")
    parser.add_argument('-l', '--label', type=str, default=None, help="label associated with the collected data")
    parser.add_argument('-p', '--allow_pickle', action='store_false', help="allow pickle numpy argument")

    # Arguments retrieving
    args = parser.parse_args()
    # -- Main args
    modelpath = args.model.strip()
    x_filepath = args.x_path.strip()
    y_filepath = args.y_path.strip()
    id_filepath = args.identifiant.strip()
    output_dir = args.output_dir.strip()
    label = args.label.strip()
    allow_pickle = args.allow_pickle

    # Arguments checking
    if not auxiliary.isfile(modelpath):
        raise Exception(f"Path to model file is incorrect :\n\t{modelpath}")

    if not auxiliary.isfile(x_filepath):
        raise Exception("Path to the data file is invalid")

    if not auxiliary.isdir(output_dir):
        raise Exception("Output directory is invalid")

    test_model(
        model.load_model(modelpath),
        x=auxiliary.load_npy(x_filepath, allow_pickle=allow_pickle),
        y=auxiliary.load_npy(y_filepath, allow_pickle=allow_pickle),
        id=auxiliary.load_npy(id_filepath, allow_pickle=allow_pickle),
        model_name=auxiliary.replace_extension(os.path.basename(modelpath), ""),
        save_to=output_dir,
        lab=label,
        metric="mse"

    )
