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
    # If no existing file to the path exists
    # ,we return the actual path
    if not os.path.isfile(filepath):
        return filepath
    # Else we append a suffix
    filename = os.path.basename(filepath)
    root, ext = os.path.splitext(filename)
    pattern_file = re.compile(root + suffix_sep + "[0-9]+" + "[.]" + ext)
    pattern_begin = re.compile("\d")
    pattern_end = re.compile("\.")
    matched_file = [pattern_file.search(filename).group(0) for filename in filename]
    matched_suffix = [filename[pattern_begin.search(filename):pattern_end.search(filename)] \
                     for filename in matched_file]
    matched_suffix = [int(number) for number in matched_suffix]
    max_suffix = max(matched_suffix)
    f""

    pass