""""""

import os
import argparse

SELECT_OPTIMIZERS = {
    "adam": "",
    "rmsprop": "",
    "sgd": ""
}
def train_model(model, X, Y, X_val=None, Y_val=None,
                epochs=10, batch_size=256, 
                saveto=None, **kwargs
):
    """"""
    XY_val = None if X_val is None and Y_val is None else \
            (X_val, Y_val)

    history = model.fit(X, Y, validation_data=XY_val, batch_size=batch_size, epochs=epochs, **kwargs)
    if saveto is not None:
        if os.path.isdir(saveto):
            model.save(saveto)
    
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
    parser.add_argument('x_val', nargs='?', type=str, default="./data/X_val.npy", help="File path to X validation file.")
    parser.add_argument('y_val', nargs='?', type=str, default="./data/Y_val.npy", help="File path to Y validation file.")
    # Output paths
    parser.add_argument('-o', '--output', type=str, default="./out/", help="output directory")
    parser.add_argument('-mo', '--model_output', type=str, default=None, help="output directory for the model")
    parser.add_argument('-w', '--model_write', action='store_false', help="should we overwrite model after training ?")
    # Fitting parameters
    parser.add_argument('-e', '--epochs', type=int, default=1, help="number of epochs to perform")
    parser.add_argument('-b', '--batch', type=int, default=256, help="batch size during training")
    # Optimizer args (optional)
    parser.add_argument('-opt', '--optimizer', type=str.lower, default=None, choices=SELECT_OPTIMIZERS.keys(), help="chosen optimizer")
    parser.add_argument('-lr', '--learning_rate', type=float, default=None, help="chosen learning rate (if opt is set)")
    parser.add_argument('-lf', '--loss_function', type=float, default=None, help="chosen loss function (if opt is set)")
    # Regularization args (optional)
    parser.add_argument('-1', '--regl1', type=float, default=0.0, help="l1 regularization factor")
    parser.add_argument('-2', '--regl2', type=float, default=0.0, help="l2 regularization factor")
    parser.add_argument('-d', '--dropout', type=float, default=0.0, help="dropout regularization value")

    #parser.add_argument('-b', '--random', action='store_true', default=False, help="Should the conformation be linearly initialized or randomly ?")
    args = parser.parse_args()
    print(args.optimizer)
