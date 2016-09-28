import pytest
import numpy as np
import plenopy as plp

class TestLixelRays(object):

    def test_init(self):

        N = 100
        np.random.seed(0)
        support = np.zeros(shape=(N,3))
        support[:,0] = np.random.rand(N)
        support[:,1] = np.random.rand(N)
        support[:,2] = np.zeros(N)

        direction = np.zeros(shape=(N,3))
        cx = 0.1*np.random.rand(N)
        cy = 0.1*np.random.rand(N)
        direction[:,0] = cx
        direction[:,1] = cy
        direction[:,2] = np.sqrt(1.0 - cx**2 - cy**2)

        x = support[:,0]
        y = support[:,1]

        rays = plp.LixelRays(
            x=support[:,0], 
            y=support[:,1],
            cx=cx,
            cy=cy)

        assert np.allclose(rays.support, support)
        assert np.allclose(rays.direction, direction)