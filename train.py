""""""

import os
import argparse

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



if _name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog = "Train.py",
            description = "This program applies a training on the chosen model."
            epilog = ""
            )

    parser.add_argument('model', type=str, help="File path to saved model file.")
    parser.add_argument('x_train', nargs='?', type=str, default="./data/X_train.npy", help="File path to X train file.")
    parser.add_argument('y_train', nargs='?', type=str, default="./data/Y_train.npy", help="File path to Y train file.")
    parser.add_argument('x_val', nargs='?', type=str, default="./data/X_val.npy", help="File path to X validation file.")
    parser.add_argument('y_val', nargs='?', type=str, default="./data/Y_val.npy", help="File path to Y validation file.")
    #parser.add_argument('-o', '--output', type=str, default="../out/", help="output directory")
    #parser.add_argument('-n', '--replica', type=int, default=3, help="Replica number (please don't exagerate)")
    #parser.add_argument('-b', '--random', action='store_true', default=False, help="Should the conformation be linearly initialized or randomly ?")
    pass