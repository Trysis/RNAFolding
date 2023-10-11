import keras

# Local modules
import auxiliary as aux


def simple_lstm(**kwargs):
    model = keras.Sequential([
        keras.layers.Input(shape=(457, 4)),
        keras.layers.LSTM(units=8, return_sequences=True),
        keras.layers.Dense(units=2, activation='linear')
    ])
    return model


def set_optimizer(model, optimizer, loss):
    """Set the chosen optimizer on the model."""
    model.compile(optimizer=optimizer, loss=loss)


def save_model(model, dirpath="", filename="model", ext="keras"):
    filename_ext = aux.replace_extension(filename, ext)
    model.save(filepath)