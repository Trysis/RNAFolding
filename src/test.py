"""This script will be used to test the model on an X data."""

import argparse

import numpy as np
import pandas as pd

import keras

# Local modules
import model
import plots
import auxiliary

def test_model(model, x, y, id=None, lab=None, metric="mse", save_to=None, kaggle_format=True):
    if save_to is not None and auxiliary.isdir(save_to):
        save_to = auxiliary.to_dirpath(save_to)
    else:
        raise Exception("Output directory is incorrect")

    y_obs = y
    y_pred = model.predict(x)
    xlabel = "valeur observée"
    ylabel = "valeur prédite"
    if id is not None:
        print(f"\n{id}\n")
        # Id
        for id, r_obs, r_pred in zip(id, y_obs, y_pred):
            to_title = "" if lab is None else f"\n{lab}"
            title_2A3 = f"{id} - 2A3" + to_title
            title_DMS = f"{id} - DMS" + to_title
            print(id)
            indices = np.arange(id.shape[0])

            # Plot 2A3
            plots.plot(indices, r_obs[:, 0], r_pred[:, 0],
                       title=title_2A3, metric=metric,
                       xlabel=xlabel, ylabel=ylabel,
                       filename=f"{id}_2A3", save_to=save_to)

            # Plot DMS
            plots.plot(indices, r_obs[:, 1], r_pred[:, 1],
                       title=title_DMS, metric=metric,
                       xlabel=xlabel, ylabel=ylabel,
                       filename=f"{id}_DMS", save_to=save_to)
    else:
        # Without ID
        for r_obs, r_pred in zip(y_obs, y_pred):
            to_title = "" if lab is None else f"\n{lab}"
            title_2A3 = f"2A3" + to_title
            title_DMS = f"DMS" + to_title
            indices = np.arange(r_obs.shape[0])

            # Plot 2A3
            plots.plot(indices, r_obs[:, 0], r_pred[:, 0],
                       title=title_2A3, metric=metric,
                       xlabel=xlabel, ylabel=ylabel,
                       filename=f"2A3", save_to=save_to)

            # Plot DMS
            plots.plot(indices, r_obs[:, 1], r_pred[:, 1],
                       title=title_DMS, metric=metric,
                       xlabel=xlabel, ylabel=ylabel,
                       filename=f"DMS", save_to=save_to)

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
        save_to=output_dir,
        lab=label,
        metric="mse"

    )
