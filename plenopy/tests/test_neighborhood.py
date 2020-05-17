import numpy as np
import plenopy as pl


def test_neighborhood():
    #  /\y
    #  | (0,1)3   (1,1)2
    #  |
    #  | (0,0)0   (1,0)1
    #  ---------------> x
    x = [0, 1, 1, 0]
    y = [0, 0, 1, 1]
    epsilon = 1.0
    nn = pl.tools.neighborhood(x=x, y=y, epsilon=epsilon, itself=False)
    assert nn.shape[0] == 4
    assert nn.shape[1] == 4
    # itself is False
    assert nn[0, 0] == False
    assert nn[1, 1] == False
    assert nn[2, 2] == False
    assert nn[3, 3] == False

    assert nn[0, 1] == True
    assert nn[0, 3] == True
    assert nn[0, 2] == False

    assert nn[1, 0] == True
    assert nn[1, 2] == True
    assert nn[1, 3] == False

    for ix in range(nn.shape[0]):
        for iy in range(nn.shape[1]):
            assert nn[ix, iy] == nn[iy, ix]

    nni = pl.tools.neighborhood(x=x, y=y, epsilon=epsilon, itself=True)

    # itself is True
    assert nni[0, 0] == True
    assert nni[1, 1] == True
    assert nni[2, 2] == True
    assert nni[3, 3] == True

    for ix in range(nni.shape[0]):
        for iy in range(nni.shape[1]):
            assert nni[ix, iy] == nni[iy, ix]
