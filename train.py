import os

def train_model(model, X, Y, X_val=None, Y_val=None,
                epochs=10, batch_size=256, 
                saveto=None, **kwargs
):
    XY_val = None if X_val is None and Y_val is None else \
            (X_val, Y_val)

    model.fit(X, Y, validation_data=XY_val, batch_size=batch_size, epochs=epochs, **kwargs)
    if saveto is not None:
        if os.path.isdir(saveto):
            model.save(saveto)
            
if _name__ == "__main__":
    pass