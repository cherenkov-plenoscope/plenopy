import numpy as np


class Binning(object):

    def __init__(
        self,
        cx_start=np.deg2rad(-3.5),
        cx_stop=np.deg2rad(+3.5),
        cx_num=32,
        cy_start=np.deg2rad(-3.5),
        cy_stop=np.deg2rad(3.5),
        cy_num=32,
        obj_start=1e3,
        obj_stop=25e3,
        obj_num=32
    ):

        self.cx_start = cx_start
        self.cx_stop = cx_stop
        self.cx_num = cx_num

        self.cy_start = cy_start
        self.cy_stop = cy_stop
        self.cy_num = cy_num

        self.obj_start = obj_start
        self.obj_stop = obj_stop
        self.obj_num = obj_num

        self.bins_num = (
            self.cx_num*
            self.cy_num*
            self.obj_num
        )

        self._assert_valid()
        self._add_bin_edges()
        self._add_widths()
        self._add_depth_of_field_cell_radii()
        self._add_bin_centers()

    def _assert_valid(self):
        assert self.cx_stop > self.cx_start
        assert self.cy_stop > self.cy_start
        assert self.obj_stop > self.obj_start

        assert self.cx_num > 0
        assert self.cy_num > 0
        assert self.obj_num > 0        

    def _add_bin_edges(self):
        self.cx_bin_edges = np.linspace(
            start=self.cx_start, 
            stop=self.cx_stop, 
            num=self.cx_num+1
        )
        self.cy_bin_edges = np.linspace(
            start=self.cy_start, 
            stop=self.cy_stop, 
            num=self.cy_num+1
        )
        self.obj_bin_edges = np.linspace(
            start=self.obj_start, 
            stop=self.obj_stop, 
            num=self.obj_num+1
        )

    def _add_widths(self):
        self.cx_width = self.cx_stop - self.cx_start
        self.cy_width = self.cy_stop - self.cy_start
        self.obj_width = self.obj_stop - self.obj_start

    def _add_depth_of_field_cell_radii(self):
        # Depth of Field (DoF)
        self.cx_bin_radius = 0.5*self.cx_width/self.cx_num
        self.cy_bin_radius = 0.5*self.cy_width/self.cy_num
        self.obj_bin_radius = 0.5*self.obj_width/self.obj_num        

    def _add_bin_centers(self):
        self.cx_bin_centers = (
            self.cy_bin_edges[: -1] + self.cx_bin_radius
        )
        self.cy_bin_centers = (
            self.cy_bin_edges[: -1] + self.cy_bin_radius
        )
        self.obj_bin_centers = (
            self.obj_bin_edges[: -1] + self.obj_bin_radius
        )

    def __repr__(self):
        r2d = np.rad2deg
        fov_solid_angle_sqdeg = r2d(self.cx_width)*r2d(self.cy_width)

        out = 'ImageDomainBinning('
        out += str(self.bins_num)+'bins, '
        out += str(fov_solid_angle_sqdeg)+'sqdeg x '
        out += str(self.obj_width)+'m)'
        return out
