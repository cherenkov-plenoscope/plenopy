import glob
import os
import numpy as np

def all_folders_with_digit_names_in_path(path):
    """
    Returns an array of event folders found in the input path directory.
    Every folder named with a valid integer digit is idenified as event folder.

    Parameters
    ----------
    path       path of a run directory     
    """

    files_in_run_folder = glob.glob(os.path.join(path, '*'))
    events = []
    for fi in files_in_run_folder:
        if os.path.isdir(fi) and os.path.basename(fi).isdigit():
            events.append(int(os.path.basename(fi)))
    events = np.array(events)
    events.sort()
    return events