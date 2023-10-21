""""""

import os
import argparse

import pandas as pd
from keras import backend
from tensorflow import keras

# Local modules
import model
import auxiliary

# Constantes
SELECT_OPTIMIZERS = {
    "adam": keras.optimizers.Adam,
    "rmsprop": keras.optimizers.RMSprop,
    "sgd": keras.optimizers.SGD,
}

SELECT_MODELS = {
    "simple_lstm": model.simple_lstm,
}


def load_model(model_link,
               optimizer=None, learning_rate=None, loss_fn=None,
               hidden_units=None, l1=None, l2=None, dropout=None,
               **kwargs
):
    """This function will return a model based on the
        specified file or name, and will apply the chosen
        optional parameters on the model.

    model_link: str
        Path to an existing model, or
        model name present in global variable {MODELS}
        that will point to the function creating the model

    optimizer: keras.optimizers.optimizer.Optimizer
        Selected optimizer

    learning_rate: float (optional)
        Attributed learning rate to the optimizer for
        gradient descent.

    loss_fn: keras.loss.* or str or <class 'function'> (optional)
        Loss function

    hidden_units: int (optional)
        Correspond to the number of hidden units for the
        model. This is useful when the model is chosen by
        name in the {MODELS} variable, specifying a model
        function taking as argument hidden_units for easier
        parametrization.

    l1, l2: float, float (optional)
        l1 and l2 regularization values for the model.
        This is useful when the model is chosen by
        name in the {MODELS} variable, specifying a model
        function taking as argument hidden_units for easier
        parametrization.

    dropout: float (optional)
        Dropout regularization value for the model.
        This is useful when the model is chosen by
        name in the {MODELS} variable, specifying a model
        function taking as argument hidden_units for easier
        parametrization.

    Returns: keras.model
        The chosen model with the specified parameters

    """
    # Name of the model and Model
    model_name, selected_model = "unknown.keras", None
    if auxiliary.isfile(model_link):
        # When path to model file is existant
        model = keras.saving.load_model(model_link)
        model_name = os.path.basename(model_link)
    else:
        # Create a model from zero, by selecting by name
        # the specified model in {SELECT_MODELS} global variable
        model = SELECT_MODELS.get(model_link, None)
        model = model(hidden_units=hidden_units,
                      l1=l1, l2=l2, dropout=dropout,
                      **kwargs)
        model_name = model_link + ".keras"

    # Set parameters if defined
    if optimizer is not None and loss_fn is not None:
        optimizer_fn = optimizer(**kwargs) if learning_rate is None else \
                    optimizer(learning_rate=learning_rate, **kwars)

        model.compile(optimizer=optimizer_fn, loss=loss_fn, **kwargs)
    elif optimizer is not None and learning_rate is not None:
        backend.set_value(model.optimizer.learning_rate, learning_rate)

    return model_name, model


def train_model(model_link, x_train, y_train, x_val=None, y_val=None,
                save_to=None, save_md_to=None, overwrite=True,
                epochs=None, batch_size=None, 
                optimizer=None, learning_rate=None, loss_fn=None,
                hidden_units=None, l1=None, l2=None, dropout=None,
                **kwargs
):
    """This function will return a model based on the
        specified file or name, and will apply the chosen
        optional parameters on the model.

    model_link: str
        Path to an existing model, or
        model name present in global variable {MODELS}
        that will point to the function creating the model

    x_train, y_train: np.ndarray, np.ndarray
        X and Y training sets for the model corresponding
        to the input variables for the prediction of the
        Y matrix.

    x_val, y_val: np.ndarray, np.ndarray (optional)
        X and Y validaiton sets for the model prediction
        validatinon.

    save_to: str (optional)
        Output directory to plot the model output

    save_md_to: str (optional)
        Output directory for the model

    overwrite: bool (default=True)
        Should we overwrite the model file after training ?
        If False, the new name will have the same model name
        with an added suffix

    optimizer: keras.optimizers.optimizer.Optimizer
        Selected optimizer

    learning_rate: float (optional)
        Attributed learning rate to the optimizer for
        gradient descent.

    loss_fn: keras.loss.* or str or <class 'function'> (optional)
        Loss function

    hidden_units: int (optional)
        Correspond to the number of hidden units for the
        model. This is useful when the model is chosen by
        name in the {MODELS} variable, specifying a model
        function taking as argument hidden_units for easier
        parametrization.

    l1, l2: float, float (optional)
        l1 and l2 regularization values for the model.
        This is useful when the model is chosen by
        name in the {MODELS} variable, specifying a model
        function taking as argument hidden_units for easier
        parametrization.

    dropout: float (optional)
        Dropout regularization value for the model.
        This is useful when the model is chosen by
        name in the {MODELS} variable, specifying a model
        function taking as argument hidden_units for easier
        parametrization.

    """
    model_name, model = load_model(model_link,
                                   hidden_units=hidden_units,
                                   optimizer=optimizer,
                                   learning_rate=learning_rate,
                                   loss_fn=loss_fn,
                                   l1=l1, l2=l2,
                                   dropout=dropout,
                                   **kwargs
                                   )

    save_format = kwargs.get("save_format", "tf")
    XY_val = None if X_val is None and Y_val is None else \
            (X_val, Y_val)

    history = model.fit(X, Y, validation_data=XY_val, batch_size=batch_size, epochs=epochs, **kwargs)
    if save_md_to is not None:
        if os.path.isdir(save_md_to):
            model.save(save_md_to, save_format=save_format)

    return history


if __name__ == "__main__":
    ## Parsing
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
    parser.add_argument('-e', '--epochs', type=int, default=10, help="number of epochs to perform")
    parser.add_argument('-b', '--batch', type=int, default=256, help="batch size during training")
    # Optimizer args (optional)
    parser.add_argument('-opt', '--optimizer', type=str.lower, default=None, choices=SELECT_OPTIMIZERS.keys(), help="chosen optimizer")
    parser.add_argument('-lr', '--learning_rate', type=float, default=None, help="chosen learning rate (if opt is set)")
    parser.add_argument('-lf', '--loss_function', type=float, default=None, help="chosen loss function (if opt is set)")
    # Node size args for model taking these inputs (optional)
    parser.add_argument('-h', '--hidden_units', type=int, default=None, help="Hidden units for a model taking this arg")
    # Regularization args (optional)
    parser.add_argument('-1', '--regl1', type=float, default=0.0, help="l1 regularization factor")
    parser.add_argument('-2', '--regl2', type=float, default=0.0, help="l2 regularization factor")
    parser.add_argument('-d', '--dropout', type=float, default=0.0, help="dropout regularization value")
    # Output paths
    parser.add_argument('-o', '--output', type=str, default="./out/", help="output directory")
    parser.add_argument('-mo', '--model_output', type=str, default="./out/", help="output directory for the model")
    parser.add_argument('-w', '--model_write', action='store_false', help="should we overwrite model after training ?")
    # Model name
    parser.add_argument('-mn', '--model_name', type=str, default="unknown", help="chosen name for the model")

    # Arguments retrieving
    args = parser.parse_args()
    # -- Input data
    model_name = args.model_name
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

    # Arguments checking
    if not auxiliary.isfile(x_train_path):
        raise Exception(f"Path to x_train is incorrect :\n\t{x_train_path}")

    if not auxiliary.isfile(y_train_path):
        raise Exception(f"Path to y_train is incorrect :\n\t{y_train_path}")

    if (x_val_path is None) or (y_val_path is None):
        if x_val_path is None:
            if y_val_path != x_val_path:
                raise Exception("If y_val_path is set, x_val_path should also be set\n"
                                f"x_val_path={x_val_path}\n"
                                f"y_val_path={y_val_path}\n"
                                )
        else:
            raise Exception("If x_val_path is set, y_val_path should also be set\n"
                            f"x_val_path={x_val_path}\n"
                            f"y_val_path={y_val_path}\n"
                            )

    if epochs <= 0:
        raise Exception("epochs value should be positive (>=0)")
    if batch_size < 1:
        batch_size = None
    if optimizer is None:
        loss_fn = None
    if l1 < 0:
        raise Exception("l1 regularization should be positive")
    if l2 < 0:
        raise Exception("l2 regularization should be positive")
    if dropout < 0:
        raise Exception("dropout value should be positive")
    if not auxiliary.isdir(output_model_dir):
        raise Exception("Ouput directory for the model is invalid")

    train_model(model_link,
                *auxiliary.load_data(x_train_path, y_train_path),
                *auxiliary.load_data(x_val_path, y_val_path),
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