import numpy as np
import plenopy as pl


def test_depth_of_field_transformations():
    FOCAL_LENGTH = 4.889
    for x in np.linspace(-1e3, 1e3, 13):
        for y in np.linspace(-1e3, 1e3, 13):
            for z in np.linspace(.1e3, 10e3, 13):
                cxcyb = pl.tomography.image_domain.transform.xyz2cxcyb(
                    x,
                    y,
                    z,
                    FOCAL_LENGTH
                )

                xyz = pl.tomography.image_domain.transform.cxcyb2xyz(
                    cxcyb[0], cxcyb[1], cxcyb[2],
                    FOCAL_LENGTH
                )

                assert np.isclose(xyz[0], x, atol=0.01)
                assert np.isclose(xyz[1], y, atol=0.01)
                assert np.isclose(xyz[2], z, atol=0.01)
