import os

# Pattern matching
import re

def replace_extension(name, ext):
    """Takes a name and replace the existing extension
        by a specified extension. Or simply add the specified
        extension.
    
    name: str
        Name of the string to add the extension to
    ext: str
        Extension value to append to {name}

    Returns: str
        The new name with {name}.{ext}
    """
    root, ext = os.path.splitext(name)
    ext.replace(".", "")
    name_ext = root + "." + ext
    return name_ext


def append_suffix(filepath, path_sep="/", suffix_sep="_"):
    """"""
    # If no existing file to the path exists
    # ,we return the actual path
    if not os.path.isfile(filepath):
        return filepath
    # Else we append a suffix
    dirname = os.path.dirname(filepath)  # directory name
    dirname = "." if dirname == "" else dirname
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

if __name__ == "__main__":
    filepath = "./data/test.py"
    filepath_with_suffix = append_suffix(filepath)
    print(filepath_with_suffix)