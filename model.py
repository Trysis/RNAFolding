import keras

# Local modules
import auxiliary as aux


def simple_lstm(input_size=(457, 4), hidden_size=32, **kwargs):
    """Simple lstm"""
    model = keras.Sequential([
        keras.layers.Input(shape=input_size),
        keras.layers.LSTM(units=hidden_size, return_sequences=True),
        keras.layers.Dense(units=2, activation='linear')
    ])
    return model


def set_optimizer(model, optimizer, loss):
    """Set the chosen optimizer on the model."""
    model.compile(optimizer=optimizer, loss=loss)
    return model


def save_model(model, dirpath="", dir_sep="/", filename="model", ext="keras"):
    if not aux.isdir(dirpath):
        raise Exception("Dirpath is invalid")

    # Save model with attributed extension
    dirpath = aux.to_dirpath(dirpath, dir_sep=dir_sep)
    filename_ext = aux.replace_extension(filename, ext)
    filepath = f"{dirpath}{filename_ext}"
    model.save(filepath)