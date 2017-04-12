import numpy as np


def time_slice_with_max_intensity(sequence):
    number_time_slices = sequence.shape[0]
    max_along_slices = np.zeros(number_time_slices, dtype=np.uint16)
    for s, slic in enumerate(sequence):
        max_along_slices[s] = slic.max()
    return np.argmax(max_along_slices)