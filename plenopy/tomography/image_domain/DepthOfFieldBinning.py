import numpy as np
from .transform import object_distance_2_image_distance as g2b
from .transform import image_distance_2_object_distance as b2g

class DepthOfFieldBinning(object):

    def __init__(
        self,
        cx_min=np.deg2rad(-3.5),
        cx_max=np.deg2rad(+3.5),
        cx_num=64,
        cy_min=np.deg2rad(-3.5),
        cy_max=np.deg2rad(3.5),
        cy_num=64,
        obj_min=1e3,
        obj_max=25e3,
        obj_num=64,
        focal_length=106.05,
        min_obj_focal_length_scale=10.0,
        max_obj_focal_length_scale=300.0,
    ):
        self.focal_length = focal_length

        self._cx_min = cx_min
        self._cx_max = cx_max
        self._cx_num = cx_num

        self._cy_min = cy_min
        self._cy_max = cy_max
        self._cy_num = cy_num

        self.x_img_min = self.focal_length*np.tan(self._cx_min)
        self.x_img_max = self.focal_length*np.tan(self._cx_max)
        self.x_img_num = self._cx_num
        self.x_img_width = self.x_img_max - self.x_img_min
        self.x_img_bin_radius = 0.5*self.x_img_width/self.x_img_num
        self.x_img_bin_edges = np.linspace(
            self.x_img_min, self.x_img_max, self.x_img_num+1
        )
        self.x_img_bin_centers = (
            self.x_img_bin_edges[: -1] + self.x_img_bin_radius
        )

        self.y_img_min = self.focal_length*np.tan(self._cy_min)
        self.y_img_max = self.focal_length*np.tan(self._cy_max)
        self.y_img_num = self._cy_num
        self.y_img_width = self.y_img_max - self.y_img_min
        self.y_img_bin_radius = 0.5*self.y_img_width/self.y_img_num
        self.y_img_bin_edges = np.linspace(
            self.y_img_min, self.y_img_max, self.y_img_num+1
        )
        self.y_img_bin_centers = (
            self.y_img_bin_edges[: -1] + self.y_img_bin_radius
        )

        self._obj_min = obj_min
        self._obj_max = obj_max
        self._obj_num = obj_num

        self.b_img_min = g2b(self._obj_max, self.focal_length) # obj MAX is b img MIN
        self.b_img_max = g2b(self._obj_min, self.focal_length)
        self.b_img_num = self._obj_num
        self.b_img_width = self.b_img_max - self.b_img_min
        self.b_img_bin_radius = 0.5*self.b_img_width/self.b_img_num
        self.b_img_bin_edges = np.linspace(
            start=self.b_img_min,
            stop=self.b_img_max,
            num=self.b_img_num+1
        )
        self.b_img_bin_centers = (
            self.b_img_bin_edges[: -1] + self.b_img_bin_radius
        )

        self.bin_num = self.x_img_num*self.y_img_num*self.b_img_num

        self._assert_valid()

    def _assert_valid(self):
        assert self.focal_length > 0.0
        assert self._cx_max > self._cx_min
        assert self._cy_max > self._cx_min
        assert self._obj_max > self._obj_min

        assert self._cx_num > 0
        assert self._cy_num > 0
        assert self._obj_num > 0

    def __repr__(self):
        out = 'DepthOfFieldBinning('
        out += str(self.x_img_num*self.y_img_num*self.b_img_num)+'bins, '
        out += str(self.x_img_width*self.y_img_width)+'m^2 x '
        out += str(self.b_img_width)+'m)'
        return out

    def xyb_voxel_positions(self):
        """
        Returns a flat array of the voxel's center positions.
        """
        x_flat = self.x_img_bin_centers.repeat(
            self.b_img_num*self.y_img_num
        )
        y_flat = np.repeat(
            np.tile(
                self.y_img_bin_centers, self.x_img_num
            ),
            self.b_img_num
        )
        b_flat = np.tile(
            self.b_img_bin_centers,
            self.x_img_num*self.y_img_num
        )
        return np.array([x_flat, y_flat, b_flat]).T


    def voxels_within_field_of_view(self, radius=1.0):
        voxel_centers = self.xyb_voxel_positions()
        voxel_dists_to_optical_axis = np.sqrt(
            voxel_centers[:,0]**2 + voxel_centers[:,1]**2
        )

        max_img_offset_xy = np.array([
            self.x_img_min,
            self.x_img_max,
            self.y_img_min,
            self.y_img_max,
        ])
        max_img_offset_xy = np.abs(max_img_offset_xy)
        max_img_offset_xy = max_img_offset_xy.mean()

        return voxel_dists_to_optical_axis < radius*max_img_offset_xy
