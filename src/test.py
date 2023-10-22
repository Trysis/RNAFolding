"""This script will be used to test the model on an X data."""

import argparse

import keras

# Local modules
import auxiliary


def test_model(modelpath, x_filepath, save_to, kaggle_format=True, csv_format=".csv"):
    model = keras.saving.load_model(model_link)
    x = auxiliary.load_npy(x_filepath)

    y_pred = model.predict(x)


if __name__ == "__main__":
    ## Parsing
    parser = argparse.ArgumentParser(
            prog = "Test.py",
            description = "This program test a model on a chosen data (X).",
            epilog = ""
            )

    # Input Data
    parser.add_argument('model', type=str, help="File path to saved model file.")
    parser.add_argument('x_path', nargs='?', type=str, default="./data/X_val.npy", help="File path to X file.")
    parser.add_argument('-o', '--output_dir', type=str, default="./out/", help="output directory")

    # Arguments retrieving
    args = parser.parse_args()
    # -- Main args
    modelpath = args.model
    x_filepath = args.x_path
    output_dir = args.output_dir

    # Arguments checking
    if not auxiliary.isfile(modelpath):
        raise Exception(f"Path to model file is incorrect :\n\t{modelpath}")

    if not auxiliary.isfile(x_filepath):
        raise Exception("Path to the data file is invalid")

    if not auxiliary.isdir(output_dir):
        raise Exception("Output directory is invalid")

    
