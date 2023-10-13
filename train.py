""""""

import os
import argparse

import pandas as pd
from tensorflow import keras

# Local modules
import model

SELECT_OPTIMIZERS = {
    "adam": keras.optimizers.Adam,
    "rmsprop": keras.optimizers.RMSprop,
    "sgd": keras.optimizers.SGD,
}

SELECT_MODELS = {
    "simple_lstm": model.simple_lstm,
}

def load_model(model_link, hidden_units=None,
               optimizer=None, learning_rate=None, loss_fn=None,
               l1=None, l2=None, dropout=None, **kwargs
):
    """This function will return a model based on the
        specified file/name.

    model: str
        Path to an existing model, or
        model name present in global variable {MODELS}

    Returns: keras.model
        The chosen model
    """
    model_name = "unknown.keras"
    selected_model = None
    if os.path.isfile(model_link):
        model = keras.saving.load_model(model_link)
        model_name = os.path.basename(model_link)
    else:
        model = SELECT_MODELS[model_link]
        model = model(hidden_units=hidden_units,
                      l1=l1, l2=l2, dropout=dropout,
                      **kwargs)
        model_name = model_link + ".keras"
    
    if optimizer is not None and loss_fn is not None:
        optimizer_fn = optimizer(**kwargs) if learning_rate is None else \
                    optimizer(learning_rate=learning_rate, **kwars)

        model.compile(optimizer=optimizer_fn, loss=loss_fn)
    
    return model_name, model


def load_data(x_path, y_path):
    """Load X and Y files from specified file paths.
    
    x_path: str
        Path to the X binary (.npy)
    y_path: str
        Path to the Y binary file (.npy)
    
    Returns: np.ndarray, np.ndarray
        X and Y numpy array
    """
    x, y = None, None
    if x_path is not None and y_path is not None:
        x_exists = os.path.isfile(x_path)
        y_exists = os.path.isfile(y_path)
        if x_exists and y_exists:
            x = np.load(x_path)
            y = np.load(y_path)
        else:
            raise Exception("One of the paths is invalid")

    return x, y

def train_model(model_link, x_train, y_train, x_val=None, y_val=None,
                epochs=10, batch_size=256, 
                optimizer=None, learning_rate=None, loss_fn=None,
                l1=None, l2=None, dropout=None,
                save_to=None, save_md_to=None, **kwargs
):
    """"""
    model_name, model = load_model(model_link)
    save_format = kwargs.get("save_format", "tf")
    XY_val = None if X_val is None and Y_val is None else \
            (X_val, Y_val)

    history = model.fit(X, Y, validation_data=XY_val, batch_size=batch_size, epochs=epochs, **kwargs)
    if save_md_to is not None:
        if os.path.isdir(save_md_to):
            model.save(save_md_to, save_format=save_format)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog = "Train.py",
            description = "This program applies a training on the chosen model.",
            epilog = ""
            )

    # Input Data
    parser.add_argument('model', type=str, help="File path to saved model file.")
    parser.add_argument('x_train', nargs='?', type=str, default="./data/X_train.npy", help="File path to X train file.")
    parser.add_argument('y_train', nargs='?', type=str, default="./data/Y_train.npy", help="File path to Y train file.")
    parser.add_argument('-xv', '--x_val', nargs='?', type=str, default="./data/X_val.npy", help="File path to X validation file.")
    parser.add_argument('-yv', '--y_val', nargs='?', type=str, default="./data/Y_val.npy", help="File path to Y validation file.")
    # Fitting parameters
    parser.add_argument('-e', '--epochs', type=int, default=None, help="number of epochs to perform")
    parser.add_argument('-b', '--batch', type=int, default=None, help="batch size during training")
    # Optimizer args (optional)
    parser.add_argument('-opt', '--optimizer', type=str.lower, default=None, choices=SELECT_OPTIMIZERS.keys(), help="chosen optimizer")
    parser.add_argument('-lr', '--learning_rate', type=float, default=None, help="chosen learning rate (if opt is set)")
    parser.add_argument('-lf', '--loss_function', type=float, default=None, help="chosen loss function (if opt is set)")
    # Regularization args (optional)
    parser.add_argument('-1', '--regl1', type=float, default=0.0, help="l1 regularization factor")
    parser.add_argument('-2', '--regl2', type=float, default=0.0, help="l2 regularization factor")
    parser.add_argument('-d', '--dropout', type=float, default=0.0, help="dropout regularization value")
    # Output paths
    parser.add_argument('-o', '--output', type=str, default="./out/", help="output directory")
    parser.add_argument('-mo', '--model_output', type=str, default=None, help="output directory for the model")
    parser.add_argument('-w', '--model_write', action='store_false', help="should we overwrite model after training ?")

    # Arguments retrieving
    args = parser.parse_args()
    # -- Input data
    model_link = args.model
    x_train_path = args.x_train
    y_train_path = args.y_train
    x_val_path = args.x_val
    y_val_path = args.y_val
    # -- Fitting parameters
    epochs = args.epochs
    batch_size = args.batch
    # -- Optimizers
    optimizer = args.optimizer
    learning_rate = args.learning_rate
    loss_fn = args.loss_function
    # -- Regularization values
    l1 = args.regl1
    l2 = args.regl2
    dropout = args.dropout
    # -- Output directory & Write attribute
    output_dir = args.output
    output_model_dir = args.model_output
    overwrite = args.model_write

    # Arguments check
    if not os.path.isfile(x_train_path):
        raise Exception(f"Path to x_train is incorrect :\n\t{x_train_path}")

    if not os.path.isfile(y_train_path):
        raise Exception(f"Path to y_train is incorrect :\n\t{y_train_path}")


    print(args.optimizer)

    train_model(model_link,
                *load_data(x_train_path, y_train_path),
                *load_data(x_val_path, y_val_path),
                epochs=epochs,
                batch_size=batch_size,
                optimizer=optimizer,
                learning_rate=learning_rate,
                loss_fn=loss_fn,
                l1=l1,
                l2=l2,
                dropout=dropout,
                save_to=output_dir,
                save_md_to=output_model_dir
                )