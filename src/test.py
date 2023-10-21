import argparse

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

    
