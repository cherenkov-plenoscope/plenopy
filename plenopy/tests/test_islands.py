import numpy as np
import plenopy as pl


def test_islands():
    idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    intensities = [0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0]
    neighborhood = [
        [1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [9, 11],
        [10, 12],
        [11, 13],
        [12, 14],
        [13],
    ]

    islands = pl.features.islands_greater_equal_threshold(
        intensities=intensities, neighborhood=neighborhood, threshold=2
    )

    assert len(islands) == 2
    assert len(islands[0]) == 3
    assert 9 in islands[0]
    assert 10 in islands[0]
    assert 11 in islands[0]
    assert len(islands[1]) == 1
    assert 4 in islands[1]

    intensities = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    islands = pl.features.islands_greater_equal_threshold(
        intensities=intensities, neighborhood=neighborhood, threshold=1
    )

    assert len(islands) == 1
    for i in idx:
        assert i in islands[0]

    intensities = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    islands = pl.features.islands_greater_equal_threshold(
        intensities=intensities, neighborhood=neighborhood, threshold=2
    )

    assert len(islands) == 0
