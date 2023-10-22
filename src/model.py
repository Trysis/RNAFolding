import keras

# Local modules
import auxiliary as aux
import loss

def simple_lstm(input_size=(457, 4), output_size=(2), to_compile=True, **kwargs):
    """Simple lstm"""
    hidden_size = kwargs.get("hidden_size")
    hidden_size = 32 if hidden_size is None else hidden_size
    # Model
    model = keras.Sequential([
        keras.layers.Input(shape=input_size),
        keras.layers.LSTM(units=hidden_size, return_sequences=True),
        keras.layers.Dense(units=output_size, activation='linear')
    ])

    # BackPropagation algorithm and lr
    if to_compile:
        set_optimizer(model=model, optimizer="adam", loss=loss.masked_loss_fn)

    return model


def bilstm(input_size=(457, 4), output_size=(2), to_compile=True, **kwargs):
    """Simple lstm"""
    # Model
    model = keras.Sequential([
        keras.layers.Input(shape=input_size),
        keras.layers.Bidirectional(
            keras.layers.LSTM(units=256, return_sequences=True)
        ),
        keras.layers.LSTM(units=128, return_sequences=True),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dense(units=32, activation='relu'),
        keras.layers.Dense(units=output_size, activation='linear')
    ])

    # BackPropagation algorithm and lr
    if to_compile:
        set_optimizer(model=model, optimizer="adam", loss=loss.masked_loss_fn)

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
