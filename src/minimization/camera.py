"""This script performs operations related to a camera"""

import numpy as np


class Camera():
    """Camera"""

    def __init__(self, height, calib_const=815):
        self.h = height
        self.calib_const = calib_const

    def measure(self, points):
        """measure"""
        x = points[0, :, :]
        y = points[1, :, :]

        alpha_feature = np.arctan2(x, self.h - y)

        return alpha_feature.max(1)

    def pixel_to_alpha(self, pixel_pos, calib_const=-1):
        """measure"""
        if calib_const == -1:
            calib_const = self.calib_const
        return np.arctan2(pixel_pos, calib_const)
