import numpy as np
import plenopy as pl


def test_array_chnunking():
    chunking = pl.plot.xyzI._start_and_end_slices_for_1D_array_chunking

    (s,e) = chunking(number_of_chunks=10, array_length=10)
    assert len(s) == 10
    assert len(e) == 10
    for i in range(len(s)):
        assert s[i] < e[i]

    (s,e) = chunking(number_of_chunks=10, array_length=100)
    assert len(s) == 10
    assert len(e) == 10
    for i in range(len(s)):
        assert s[i] < e[i]