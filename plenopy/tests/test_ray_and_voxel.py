import pytest
import numpy as np
import plenopy as pl

overlap_of_ray_with_voxels = [
    pl.tomography.ray_and_voxel._py_overlap.overlap_of_ray_with_voxels,
    pl.tomography.ray_and_voxel.cython_overlap.overlap_of_ray_with_voxels,
]

single_voxel_overlap = [
    pl.tomography.ray_and_voxel._py_overlap._ray_box_overlap,
    pl.tomography.ray_and_voxel.cython_overlap.ray_box_overlap
]

def test_plane_intersection():

    dim = 0

    isec = pl.tomography.ray_and_voxel._py_overlap._intersection_plane(
        support=np.array([0.,0.,0.]), 
        direction=np.array([1.,0.,0.]), 
        off=0.0,
        dim=dim
    )
    assert len(isec) == 3
    assert isec[0] == 0.0
    assert isec[1] == 0.0
    assert isec[2] == 0.0

    isec = pl.tomography.ray_and_voxel._py_overlap._intersection_plane(
        support=np.array([0.,0.,0.]), 
        direction=np.array([1.,0.,1.]), 
        off=0.0,
        dim=dim
    )
    assert len(isec) == 3
    assert isec[0] == 0.0
    assert isec[1] == 0.0
    assert isec[2] == 0.0

    isec = pl.tomography.ray_and_voxel._py_overlap._intersection_plane(
        support=np.array([0.,0.,0.]), 
        direction=np.array([1.,0.,1.]), 
        off=1.0,
        dim=dim
    )
    assert len(isec) == 3
    assert isec[0] == 1.0
    assert isec[1] == 0.0
    assert isec[2] == 1.0


def ray_and_empty_voxel_overlap():
    for func in single_voxel_overlap:
    
        for xs in np.linspace(-1.,1.,3):
            for ys in np.linspace(-1.,1.,3):
                for zs in np.linspace(-1.,1.,3):
                    for xd in np.linspace(-1.,1.,3):
                        for yd in np.linspace(-1.,1.,3):
                            for zd in np.linspace(-1.,1.,3):
                                if xd!=0. or yd!=0. or zd!=0.:
                                    ol = func(
                                        support=np.array([xs,ys,zs]), 
                                        direction=np.array([xd,yd,zd]), 
                                        xl=0., xu=0., 
                                        yl=0., yu=0., 
                                        zl=0., zu=0.
                                    )

                                    assert ol == 0.0


def test_ray_and_single_voxel_overlap():
    for func in single_voxel_overlap:
        for dim in range(3):
            direction = np.zeros(3)
            direction[dim] = 1.0

            ol = func(
                support=np.array([0.,0.,0.]), 
                direction=direction, 
                xl=-1, xu=1, 
                yl=-1, yu=1, 
                zl=-1, zu=1
            )
            assert ol == 2.0



def test_overlap_single_box_edge_cases():
    for func in single_voxel_overlap:
        # along X-axis
        # ------------
        direction = np.array([1.,0.,0.])
        # z low
        ol = func(
            support=np.array([0.,0.,-1.]), 
            direction=direction, 
            xl=-1, xu=1, 
            yl=-1, yu=1, 
            zl=-1, zu=1
        )
        assert ol == 2.0

        # z upper
        ol = func(
            support=np.array([0.,0.,1.]), 
            direction=direction, 
            xl=-1, xu=1, 
            yl=-1, yu=1, 
            zl=-1, zu=1
        )
        assert ol == 0.0

        # z low
        ol = func(
            support=np.array([0.,0.,-1.]), 
            direction=direction, 
            xl=-1, xu=1, 
            yl=-1, yu=1, 
            zl=-1, zu=1
        )
        assert ol == 2.0

        # z upper
        ol = func(
            support=np.array([0.,0,1]), 
            direction=direction, 
            xl=-1, xu=1, 
            yl=-1, yu=1, 
            zl=-1, zu=1
        )
        assert ol == 0.0


        # along Y-axis
        # ------------
        direction = np.array([0.,1.,0.])

        # x low
        ol = func(
            support=np.array([-1.,0.,0.]), 
            direction=direction, 
            xl=-1, xu=1, 
            yl=-1, yu=1, 
            zl=-1, zu=1
        )
        assert ol == 2.0

        # x upper
        ol = func(
            support=np.array([1.,0.,0.]), 
            direction=direction, 
            xl=-1, xu=1, 
            yl=-1, yu=1, 
            zl=-1, zu=1
        )
        assert ol == 0.0

        # z low
        ol = func(
            support=np.array([0.,0.,-1.]), 
            direction=direction, 
            xl=-1, xu=1, 
            yl=-1, yu=1, 
            zl=-1, zu=1
        )
        assert ol == 2.0

        # z upper
        ol = func(
            support=np.array([0.,0.,1.]), 
            direction=direction, 
            xl=-1, xu=1, 
            yl=-1, yu=1, 
            zl=-1, zu=1
        )
        assert ol == 0.0

        # along Z-axis
        # ------------
        direction = np.array([0.,0.,1.])

        # x low
        ol = func(
            support=np.array([-1.,0.,0.]), 
            direction=direction, 
            xl=-1, xu=1, 
            yl=-1, yu=1, 
            zl=-1, zu=1
        )
        assert ol == 2.0

        # x upper
        ol = func(
            support=np.array([1.,0.,0.]), 
            direction=direction, 
            xl=-1, xu=1, 
            yl=-1, yu=1, 
            zl=-1, zu=1
        )
        assert ol == 0.0

        # y low
        ol = func(
            support=np.array([0.,-1.,0.]), 
            direction=direction, 
            xl=-1, xu=1, 
            yl=-1, yu=1, 
            zl=-1, zu=1
        )
        assert ol == 2.0

        # y upper
        ol = func(
            support=np.array([0.,1.,0.]), 
            direction=direction, 
            xl=-1, xu=1, 
            yl=-1, yu=1, 
            zl=-1, zu=1
        )
        assert ol == 0.0


def test_overlap_single_box_various_supports():
    for func in single_voxel_overlap:

        for dim in range(3):
            direction = np.zeros(3)
            direction[dim] = 1.0
            for xs in np.linspace(-.5,.5,3):
                for ys in np.linspace(-.5,.5,3):
                    for zs in np.linspace(-.5,.5,3):
                        ol = func(
                            support=np.array([xs,ys,zs]), 
                            direction=direction, 
                            xl=-1, xu=1, 
                            yl=-1, yu=1, 
                            zl=-1, zu=1
                        )
                        assert ol == 2.0


def test_overlap_single_box_diagonal_rays():
    for func in single_voxel_overlap:

        ol = func(
            support=np.array([0.,0.,0.]), 
            direction=np.array([-1.,-1.,-1.]), 
            xl=-1, xu=1, 
            yl=-1, yu=1, 
            zl=-1, zu=1
        )
        assert ol > 0.0



def test_overlap_single_box_various_rays():
    for func in single_voxel_overlap:

        max_diagonal = np.linalg.norm(np.array([2,2,2]))
        for xs in np.linspace(-.5,.5,3):
            for ys in np.linspace(-.5,.5,3):
                for zs in np.linspace(-.5,.5,3):
                    for xd in np.linspace(-1.,1.,3):
                        for yd in np.linspace(-1.,1.,3):
                            for zd in np.linspace(-1.,1.,3):
                                if xd!=0 or yd!=0 or zd!=0:
                                    #print('----------------------------------')
                                    ol = func(
                                        support=np.array([xs,ys,zs]), 
                                        direction=np.array([xd,yd,zd]), 
                                        xl=-1, xu=1, 
                                        yl=-1, yu=1, 
                                        zl=-1, zu=1
                                    )
                                    #print('[',xs,ys,zs,'][',xd,yd,zd,']', ol)
                                    assert ol > 0.0 and ol <= max_diagonal
                                    

def test_bin_index_space_partitioning():
    nsp = pl.tomography.ray_and_voxel._py_overlap._next_space_partitions

    s = nsp([0,0])
    assert len(s) == 1
    assert s[0][0] == 0
    assert s[0][1] == 0

    s = nsp([0,1])
    assert len(s) == 1
    assert s[0][0] == 0
    assert s[0][1] == 1

    s = nsp([1,1])
    assert len(s) == 1
    assert s[0][0] == 1
    assert s[0][1] == 1

    s = nsp([0,63])
    assert len(s) == 2
    assert s[0][0] == 0
    assert s[0][1] == 31
    assert s[1][0] == 31
    assert s[1][1] == 63

    s = nsp([0,27])
    assert len(s) == 2
    assert s[0][0] == 0
    assert s[0][1] == 13
    assert s[1][0] == 13
    assert s[1][1] == 27

    s = nsp([50,63])
    assert len(s) == 2
    assert s[0][0] == 50
    assert s[0][1] == 56
    assert s[1][0] == 56
    assert s[1][1] == 63

    s = nsp([62,63])
    assert len(s) == 1
    assert s[0][0] == 62
    assert s[0][1] == 63

    s = nsp([63,63])
    assert len(s) == 1
    assert s[0][0] == 63
    assert s[0][1] == 63



def test_overlap_in_octtree_space_edge_cases():
    for func in overlap_of_ray_with_voxels:

        # lower x-edge
        # ------------
        ol = func(
            support=np.array([-1.,0.,0.]), 
            direction=np.array([0.,0.,1.]), 
            x_bin_edges=np.linspace(-1.,1.,3), 
            y_bin_edges=np.linspace(-1.,1.,3), 
            z_bin_edges=np.linspace(-1.,1.,3)
        )
        assert len(ol['x']) == 2
        assert len(ol['y']) == 2
        assert len(ol['z']) == 2
        assert len(ol['overlap']) == 2
        assert ol['x'][0] == 0
        assert ol['x'][1] == 0
        assert ol['y'][0] == 1
        assert ol['y'][1] == 1
        assert ol['z'][0] == 0
        assert ol['z'][1] == 1
        assert ol['overlap'][0] == 1.0
        assert ol['overlap'][1] == 1.0


        # valid 1st x-bin
        # ---------------
        ol = func(
            support=np.array([-.5,0,0]), 
            direction=np.array([0.,0.,1.]), 
            x_bin_edges=np.linspace(-1,1,3), 
            y_bin_edges=np.linspace(-1,1,3), 
            z_bin_edges=np.linspace(-1,1,3)
        )
        assert len(ol['x']) == 2
        assert len(ol['y']) == 2
        assert len(ol['z']) == 2
        assert len(ol['overlap']) == 2
        assert ol['x'][0] == 0
        assert ol['x'][1] == 0
        assert ol['y'][0] == 1
        assert ol['y'][1] == 1
        assert ol['z'][0] == 0
        assert ol['z'][1] == 1
        assert ol['overlap'][0] == 1.0
        assert ol['overlap'][1] == 1.0


        # x-center
        # --------
        ol = func(
            support=np.array([0.,0.,0.]), 
            direction=np.array([0.,0.,1.]), 
            x_bin_edges=np.linspace(-1.,1.,3), 
            y_bin_edges=np.linspace(-1.,1.,3), 
            z_bin_edges=np.linspace(-1.,1.,3)
        )
        assert len(ol['x']) == 2
        assert len(ol['y']) == 2
        assert len(ol['z']) == 2
        assert len(ol['overlap']) == 2
        assert ol['x'][0] == 1
        assert ol['x'][1] == 1
        assert ol['y'][0] == 1
        assert ol['y'][1] == 1
        assert ol['z'][0] == 0
        assert ol['z'][1] == 1
        assert ol['overlap'][0] == 1.0
        assert ol['overlap'][1] == 1.0

        # valid 2nd x-bin
        # ---------------
        ol = func(
            support=np.array([.5,0,0]), 
            direction=np.array([0.,0.,1.]), 
            x_bin_edges=np.linspace(-1.,1.,3), 
            y_bin_edges=np.linspace(-1.,1.,3), 
            z_bin_edges=np.linspace(-1.,1.,3)
        )
        assert len(ol['x']) == 2
        assert len(ol['y']) == 2
        assert len(ol['z']) == 2
        assert len(ol['overlap']) == 2
        assert ol['x'][0] == 1
        assert ol['x'][1] == 1
        assert ol['y'][0] == 1
        assert ol['y'][1] == 1
        assert ol['z'][0] == 0
        assert ol['z'][1] == 1
        assert ol['overlap'][0] == 1.0
        assert ol['overlap'][1] == 1.0

        # upper x-edge
        # ------------
        ol = func(
            support=np.array([1.,0.,0.]), 
            direction=np.array([0.,0.,1.]), 
            x_bin_edges=np.linspace(-1.,1.,3), 
            y_bin_edges=np.linspace(-1.,1.,3), 
            z_bin_edges=np.linspace(-1.,1.,3)
        )
        assert len(ol['x']) == 0
        assert len(ol['y']) == 0
        assert len(ol['z']) == 0
        assert len(ol['overlap']) == 0


def test_non_straight_overlaps():
    for func in overlap_of_ray_with_voxels:

        ol = func(
            support=np.array([-1.,-1.,-1.]), 
            direction=np.array([1.,1.,1.]), 
            x_bin_edges=np.linspace(-1.,1.,3), 
            y_bin_edges=np.linspace(-1.,1.,3), 
            z_bin_edges=np.linspace(-1.,1.,3)
        )
        assert len(ol['x']) == 2
        assert len(ol['y']) == 2
        assert len(ol['z']) == 2
        assert len(ol['overlap']) == 2
        assert ol['x'][0] == 0
        assert ol['x'][1] == 1
        assert ol['y'][0] == 0
        assert ol['y'][1] == 1
        assert ol['z'][0] == 0
        assert ol['z'][1] == 1
        assert np.abs(ol['overlap'][0] - np.linalg.norm(np.array([1,1,1]))) < 1e-6
        assert np.abs(ol['overlap'][1] - np.linalg.norm(np.array([1,1,1]))) < 1e-6


def test_non_linear_voxel_space():
    for func in overlap_of_ray_with_voxels:

        ol = func(
            support=np.array([0.,.1,0.]), 
            direction=np.array([.1,.5,1.]), 
            x_bin_edges=np.logspace(0.,1.,10), 
            y_bin_edges=np.logspace(0.,1.,10), 
            z_bin_edges=np.logspace(0.,1.,10)
        )


def test_tomographic_PSF():

    np.random.seed(0)

    N_RAYS = 100
    supports = np.array([
        np.random.uniform(-2.5, 2.5, N_RAYS),
        np.random.uniform(-2.5, 2.5, N_RAYS),
        np.zeros(100)
    ]).T

    directions = np.array([
        np.random.uniform(-0.3, 0.3, N_RAYS),
        np.random.uniform(-0.3, 0.3, N_RAYS),
        np.ones(100)
    ]).T

    N_X_BINS = 8
    N_Y_BINS = 8
    N_Z_BINS = 8
    psf = pl.tomography.ray_and_voxel.point_spread_function(
        supports=supports, 
        directions=directions, 
        x_bin_edges=np.linspace(-100., 100., N_X_BINS+1), 
        y_bin_edges=np.linspace(-100., 100., N_Y_BINS+1),
        z_bin_edges=np.linspace(0., 200., N_Z_BINS+1),
    )

    # crazy prime number binning
    """
    N_X_BINS = 47
    N_Y_BINS = 59
    N_Z_BINS = 67
    psf = pl.tomography.ray_and_voxel.point_spread_function(
        supports=supports, 
        directions=directions, 
        x_bin_edges=np.linspace(-100., 100., N_X_BINS+1), 
        y_bin_edges=np.linspace(-100., 100., N_Y_BINS+1),
        z_bin_edges=np.linspace(0., 200., N_Z_BINS+1),
    )
    """
    # SEG FAULT DEATH TRAP