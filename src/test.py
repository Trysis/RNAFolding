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
               model_name="unknown", lab=None, metric="mse",
               save_to=None, kaggle_format=True, overwrite=False
):
    """"""
    if save_to is not None and auxiliary.isdir(save_to):
        save_to = auxiliary.to_dirpath(save_to)
    else:
        raise Exception("Output directory is incorrect")

    y_obs = y
    y_pred = model.predict(x)
    xlabel = "valeur observée"
    ylabel = "valeur prédite"
    
    save_to = auxiliary.create_dir(save_to + model_name, add_suffix=overwrite)
    to_title = f"\n{model_name}" if lab is None else f"\n{model_name}\n{lab}"
    if id is not None:
        # Id
        for id_seq, r_obs, r_pred in zip(id, y_obs, y_pred):
            title_2A3 = f"{id_seq} - 2A3" + to_title
            title_DMS = f"{id_seq} - DMS" + to_title
            indices = np.arange(r_obs.shape[0])
            # Plot 2A3
            ## Metrics
            isnotnan = ~np.isnan(r_obs[:, 0]) & ~np.isnan(r_pred[:, 0])
            r2_2A3 = r2_score(r_obs[:, 0][isnotnan], r_pred[:, 0][isnotnan])
            plots.plot(indices, r_obs[:, 0], r_pred[:, 0],
                       title=title_2A3, metric=metric, r2=r2_2A3,
                       xlabel=xlabel, ylabel=ylabel,
                       filename=f"R2={r2_2A3:2.4f}_{id_seq}_2A3",
                       forcename=True,
                       save_to=save_to)

            plt.clf()  # Clear plot

            # Plot DMS
            ## Metrics
            isnotnan = ~np.isnan(r_obs[:, 1]) & ~np.isnan(r_pred[:, 1])
            r2_DMS = r2_score(r_obs[:, 1][isnotnan], r_pred[:, 1][isnotnan])
            plots.plot(indices, r_obs[:, 1], r_pred[:, 1],
                       title=title_DMS, metric=metric, r2=r2_DMS,
                       xlabel=xlabel, ylabel=ylabel,
                       filename=f"R2={r2_DMS:2.4f}_{id_seq}_DMS",
                       forcename=True,
                       save_to=save_to)

            plt.close()
    else:
        # Without ID
        for r_obs, r_pred in zip(y_obs, y_pred):
            to_title = "" if lab is None else f"\n{lab}"
            title_2A3 = f"2A3" + to_title
            title_DMS = f"DMS" + to_title
            indices = np.arange(r_obs.shape[0])

            # Plot 2A3
            ## Metrics
            isnotnan = ~np.isnan(r_obs[:, 0]) & ~np.isnan(r_pred[:, 0])
            r2_2A3 = r2_score(r_obs[:, 0][isnotnan], r_pred[:, 0][isnotnan])
            plots.plot(indices, r_obs[:, 0], r_pred[:, 0],
                       title=title_2A3, metric=metric, r2=r2_2A3,
                       xlabel=xlabel, ylabel=ylabel,
                       filename=f"R2={r2_2A3:.4f}_2A3",
                       forcename=True,
                       save_to=save_to)

            plt.clf()  # Clear plot

            # Plot DMS
            ## Metrics
            isnotnan = ~np.isnan(r_obs[:, 1]) & ~np.isnan(r_pred[:, 1])
            r2_DMS = r2_score(r_obs[:, 1][isnotnan], r_pred[:, 1][isnotnan])
            plots.plot(indices, r_obs[:, 1], r_pred[:, 1],
                       title=title_DMS, metric=metric, r2=r2_DMS,
                       xlabel=xlabel, ylabel=ylabel,
                       filename=f"R2={r2_DMS:.4f}_DMS",
                       forcename=True,
                       save_to=save_to)

            plt.close()


def read_test_submission(kaggle_format, save_to):
    """Does not work for the moment"""
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
    parser.add_argument('-l', '--label', type=str, default="train", help="label associated with the collected data")
    parser.add_argument('-p', '--allow_pickle', action='store_false', help="allow pickle numpy argument")

    # Arguments retrieving
    args = parser.parse_args()
    # -- Main args
    modelpath = args.model
    x_filepath = args.x_path
    y_filepath = args.y_path
    id_filepath = args.identifiant
    output_dir = args.output_dir
    label = args.label
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
