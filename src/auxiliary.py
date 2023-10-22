<<<<<<< HEAD
=======
"""Contains utilitary functions."""

>>>>>>> roude
import os

# Pattern matching
import re

import numpy as np
from sklearn.model_selection import train_test_split


def isfile(filepath):
    """Checks if {filepath} is a valid path to file."""
    return os.path.isfile(filepath)


def isdir(dirpath):
    """Checks if {dirpath} is a valid directory."""
    return os.path.isdir(dirpath)


def to_dirpath(dirpath, dir_sep="/"):
<<<<<<< HEAD
    """Returns a dirpath with its ending file separator."""
    dirpath = dirpath if filedir[-1] == dir_sep else \
=======
    """Returns a {dirpath} with its ending file separator."""
    dirpath = dirpath if dirpath[-1] == dir_sep else \
>>>>>>> roude
              dirpath + dir_sep

    return dirpath


def save_npy(x, filepath_x, *args):
    """Save a set of numpy array.
    
    x: np.ndarray
        numpy array to save
    filepath_x:
        filename of x.npy data
    args:
        even number of arguments, such that it
        contains numpy array and filepath associated
        with it. args=[y, filepath_y, z, filepath_z, ...]

    """
    to_save = [x , filepath_x]
    to_save += tuple(args) if args is not None else []
    for i in range(0, len(to_save), 2):
        filename = replace_extension(to_save[i + 1], "npy")
        np.save(filename, to_save[i])


<<<<<<< HEAD
def load_npy(x_path, y_path):
=======
def load_npy(path, **kwargs):
    """Load a npy binary file from a specified path"""
    if not isfile(path):
        return None

    return np.load(path, **kwargs)


def load_npy_xy(x_path, y_path):
>>>>>>> roude
    """Load X and Y files from specified file paths.
    
    x_path: str
        Path to the X binary file (.npy)
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
            raise Exception("One of the path is invalid")

    return x, y


def min_max(arraylike):
    """Returns the min and max from an array."""
    return min(arraylike), max(arraylike)


def min_max_normalization(values, min_scale, max_scale):
    """Normalize values on a specified min and max range.

    values: array-like (numpy.ndarray) -> shape (n_samples, x)
        Values to perform normalization on
    min_scale: float
        Bottom range limit to apply on values so that
        values range from [values.min, values.max] to values[min_scale, values.max]
    max_scale: float
        Upper range limit to apply on values so that
        values range from [values.min, values.max] to values[values.min, max_scale]

    Returns: array-like of shape (n_samples, x)
        Normalized array in range [min_scale, max_scale]

    """
    min_val, max_val = values.min(), values.max()

    # Normalization
    scale_plage = max_scale - min_scale
    val_plage = max_val - min_val
    flex_shift = values - min_val
    flex_normalized = (flex_shift * (scale_plage/val_plage)) + min_scale

    # Returns
    return flex_normalized


def replace_extension(name, new_ext):
    """Takes a name and replace the existing extension
        by a specified extension. Or simply add the specified
        extension.

    name: str
        Name of the string to add the extension to
    new_ext: str
        Extension value to append to {name}

    Returns: str
        The new name with {name}.{new_ext}

    """
    root, _ = os.path.splitext(name)
    new_ext = new_ext.replace(".", "")
    name_ext = root + "." + new_ext

    return name_ext


def append_suffix(filepath, path_sep="/", suffix_sep="_"):
    """Takes a path to a file and append a suffix on the file
        name if necessary. It is used in case we want to have
        multiple version of the same file while not removing the
        previous ones.

    filepath: str
        Path to the file
    path_sep: str
        Path character separator. It can differ
        between Linux and Windows, so it should be
        changed accordingly.
    suffix_sep: str
        Character that will separate the actual filename
        from the count value in filepath={dirpath}{filename}
        {suffix_sep}{count}{ext}

    Returns: str
        The new filepath if the file was already existant,
        else it returns filepath.

    """
    # If no existing file to the path exists
    # ,we return the actual path
    if not os.path.isfile(filepath):
        return filepath
    # Else we append a suffix
    dirname = os.path.dirname(filepath)  # directory name
    dirname = "." if dirname == "" else dirname
    dirname = to_dirpath(dirname, path_sep)
    filename = os.path.basename(filepath)  # file name
    file_no_ext, ext = os.path.splitext(filename)
    # Match files with the same pattern as ours
    to_match = file_no_ext + suffix_sep + "[0-9]+" + "[.]" + ext[1:] + "$"
    pattern_file = re.compile(to_match)
    pattern_number = re.compile("\d")
    matched_file = [pattern_file.search(filename) for filename in os.listdir(dirname)]
    matched_file = [found.group(0) for found in matched_file if found is not None]
    # Match with suffix to get the maximum value after {suffix_sep} in matched files
    matched_suffix = [filename[slice(*pattern_number.search(filename).span())] \
                     for filename in matched_file]
    matched_suffix = [0] + [int(number) for number in matched_suffix]
    max_suffix = max(matched_suffix) + 1
    filepath_suffix = f"{dirname}/{file_no_ext}{suffix_sep}{max_suffix}{ext}"

    return filepath_suffix


def train_val_test_split(x, y, train_size=0.7, val_size=0.15, seed=42):
    """Returns the train, validation and test set from a given data."""
    if train_size + val_size >= 1:
        train_size=0.7
        val_size=0.15

    # Test proportion
    test_size = 1 - train_size - val_size
    # Define train data
    x_train, x_val_test, y_train, y_val_test = train_test_split(
        x, y, test_size=1-train_size, random_state=seed
    )
    # Define val and test data
    x_val, x_test, y_val, y_test = train_test_split(
        x_val_test, y_val_test, test_size=test_size/(1-train_size), random_state=seed
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == "__main__":
    filepath = "./data/test.py"
    filepath_with_suffix = append_suffix(filepath)
    print(filepath_with_suffix)
