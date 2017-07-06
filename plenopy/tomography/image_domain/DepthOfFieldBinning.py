import numpy as np
from .transform import object_distance_2_image_distance as g2b
from .transform import image_distance_2_object_distance as b2g

class DepthOfFieldBinning(object):

    def __init__(
        self,
        cx_min=np.deg2rad(-3.5),
        cx_max=np.deg2rad(+3.5),
        cx_num=32,
        cy_min=np.deg2rad(-3.5),
        cy_max=np.deg2rad(3.5),
        cy_num=32,
        obj_min=1e3,
        obj_max=25e3,
        obj_num=32,
        focal_length=106.05,
        min_obj_focal_length_scale=3.0,
        max_obj_focal_length_scale=300.0,
    ):
        self.focal_length = focal_length

        self.cx_min = cx_min
        self.cx_max = cx_max
        self.cx_num = cx_num

        self.cy_min = cy_min
        self.cy_max = cy_max
        self.cy_num = cy_num

        self.obj_min = obj_min
        self.obj_max = obj_max
        self.obj_num = obj_num

        self.b_min = g2b(self.obj_min , self.focal_length)
        self.b_max = g2b(self.obj_max , self.focal_length)
        self.b_num = self.obj_num

        self._assert_valid()
        self._add_bin_edges()
        self._add_widths()
        self._add_depth_of_field_cell_radii()
        self._add_bin_centers()

    def _assert_valid(self):
        assert self.focal_length > 0.0
        assert self.cx_max > self.cx_min
        assert self.cy_max > self.cy_min
        assert self.obj_max > self.obj_min

        assert self.cx_num > 0
        assert self.cy_num > 0
        assert self.obj_num > 0        

    def _add_bin_edges(self):
        self.cx_bin_edges = np.linspace(
            start=self.cx_min, 
            stop=self.cx_max, 
            num=self.cx_num+1
        )
        self.cy_bin_edges = np.linspace(
            start=self.cy_min, 
            stop=self.cy_max, 
            num=self.cy_num+1
        )
        self.b_bin_edges = np.linspace(
            start=self.b_min, 
            stop=self.b_max, 
            num=self.b_num+1
        )
        self.obj_bin_edges = b2g(self.b_bin_edges, self.focal_length)

    def _add_widths(self):
        self.cx_width = self.cx_max - self.cx_min
        self.cy_width = self.cy_max - self.cy_min
        self.obj_width = self.obj_max - self.obj_min
        self.b_width = self.b_max - self.b_min

    def _add_depth_of_field_cell_radii(self):
        # Depth of Field (DoF)
        self.cx_bin_radius = 0.5*self.cx_width/self.cx_num
        self.cy_bin_radius = 0.5*self.cy_width/self.cy_num
        self.b_bin_radius = 0.5*self.b_width/self.b_num        

    def _add_bin_centers(self):
        self.cx_bin_centers = (
            self.cy_bin_edges[: -1] + self.cx_bin_radius
        )
        self.cy_bin_centers = (
            self.cy_bin_edges[: -1] + self.cy_bin_radius
        )
        self.b_bin_centers = (
            self.b_bin_edges[: -1] + self.b_bin_radius
        )
        self.obj_bin_centers = b2g(self.b_bin_centers, self.focal_length)

    def __repr__(self):
        r2d = np.rad2deg
        fov_solid_angle_sqdeg = r2d(self.cx_width)*r2d(self.cy_width)

        out = 'DepthOfFieldBinning('
        out += str(self.bins_num)+'bins, '
        out += str(fov_solid_angle_sqdeg)+'deg^2 x '
        out += str(self.b_width)+'m)'
        return out
     
