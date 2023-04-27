import os
import glob
import numpy as np
import gzip
import subprocess


def all_folders_with_digit_names_in_path(path):
    """
    Returns an array of event folders found in the input path directory.
    Every folder named with a valid integer digit is idenified as event folder.

    Parameters
    ----------
    path       path of a run directory
    """
    files_in_run_folder = glob.glob(os.path.join(path, "*"))
    events = []
    for fi in files_in_run_folder:
        if os.path.isdir(fi) and os.path.basename(fi).isdigit():
            events.append(int(os.path.basename(fi)))
    events = np.array(events, dtype=np.int64)
    events.sort()
    return events


def is_gzipped(path):
    """
    Check for gzip file, see https://tools.ietf.org/html/rfc1952#page-5

    Reads in the first two bytes of a file and compares with the gzip magic
    numbers.
    """
    with open(path, "rb") as fin:
        marker = fin.read(2)
        if len(marker) < 2:
            return False
        return marker[0] == 31 and marker[1] == 139


def compress_event_in_place(event_path):
    phs_path = os.path.join(event_path, "raw_light_field_sensor_response.phs")
    if os.path.isfile(phs_path):
        subprocess.check_call(["gzip", phs_path])
    pulse_origins_path = os.path.join(
        event_path, "simulation_truth", "detector_pulse_origins.bin"
    )
    if os.path.isfile(pulse_origins_path):
        subprocess.check_call(["gzip", pulse_origins_path])


class gz_transparent_open:
    def __init__(self, path, file_options):
        self.file_options = file_options
        if os.path.isfile(path):
            self.path = path
        elif os.path.isfile(path + ".gz"):
            self.path = path + ".gz"

    def __enter__(self):
        if is_gzipped(self.path):
            self.fd = gzip.open(self.path, self.file_options)
        else:
            self.fd = open(self.path, self.file_options)
        return self.fd

    def __exit__(self, exc_type, exc_value, traceback):
        self.fd.close()
