import array
import numpy as np


def list_of_lists_to_arrays(list_of_lists):
    starts = array.array("l")
    lengths = array.array("l")
    stream = array.array("l")
    i = 0
    for _list in list_of_lists:
        starts.append(i)
        length = 0
        for symbol in _list:
            stream.append(symbol)
            i += 1
            length += 1
        lengths.append(length)
    return {
        "starts": np.array(starts, dtype=np.uint32),
        "lengths": np.array(lengths, dtype=np.uint32),
        "links": np.array(stream, dtype=np.uint32),
    }


def arrays_to_list_of_lists(starts, lengths, links):
    number_lixel = starts.shape[0]
    lol = [[] for p in range(number_lixel)]
    for lixel in range(number_lixel):
        for l in range(lengths[lixel]):
            idx = starts[lixel] + l
            lol[lixel].append(links[idx])
    return lol
