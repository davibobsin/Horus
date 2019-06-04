"""This script performs operations related to a camera"""

from os.path import exists
import cv2
import numpy as np


class Camera():
    """Camera"""

    def __init__(self, height, calib_const=815):
        self.h = height
        self.calib_const = calib_const

    def simulate_measurement(self, points):
        """simulate_measurement"""
        x = points[0, :, :]
        y = points[1, :, :]

        alpha_feature = np.arctan2(x, self.h - y)

        return alpha_feature.max(1)

    def measure(self, path):
        """measure"""
        x = []
        y = []
        for i in range(360):
            full_path = path+str(i)+'.png'
            if exists(full_path):
                img = cv2.imread(full_path)
                ppi, med = self.__lims(img)
                x += [i]
                y += [med]

        theta = np.array(x)
        alpha = np.array(y)

        return np.radians(theta), alpha

    def __pixel_to_alpha(self, pixel_pos):
        """__pixel_to_alpha"""
        return np.arctan2(pixel_pos, self.calib_const)

    def __ex(self, mat):
        """__ex"""
        ret = np.zeros(mat.shape)
        rmat = mat[::-1]
        h, w = mat.shape
        soma = 0
        for y in range(h):
            vet = mat[y, :]
            rvet = vet[::-1]
            minx = np.argmax(vet)
            maxx = len(rvet)-np.argmax(rvet)
            ret[y, minx] = 255
            ret[y, maxx] = 255
            soma += maxx-minx
        med = soma/h
        return ret, self.__pixel_to_alpha(med)

    def __lims(self, img):
        """__lims"""
        edges = cv2.Canny(img, 100, 200)
        h, w = edges.shape

        ret = np.zeros((h, w, 3))
        ret[:, :, 0] = edges
        cv2.rectangle(ret, (200, 400), (500, 480), (0, 0, 255), 2)

        roi = edges[400:480, 200:500]
        r, m = self.__ex(roi)
        ret[400:480, 200:500, 1] += r
        return ret, m
