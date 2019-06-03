"""This script performs operations related to an
hypothetical blank with a given section"""

import numpy as np


class Section():
    """Section"""

    def __init__(self, points):
        self.x = points[:, 0]
        self.y = points[:, 1]

        self.r = np.sqrt(self.x**2+self.y**2)
        self.phi = np.arctan2(self.y, self.x)

    def rotate(self, theta):
        """rotate"""

        psi = np.add.outer(theta, self.phi)

        x, y = self.to_xy(self.r, psi)

        return np.array([x, y])

    def plot(self, plt):
        """plot"""

        Xs = np.append(self.x, self.x[0])
        Ys = np.append(self.y, self.y[0])

        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

        plt.plot(Xs, Ys, marker='o', color=color)
        plt.gca().set_aspect('equal', adjustable='box')

    @staticmethod
    def to_xy(r, phi, to_plot=False):
        """to_xy"""

        x = r*np.cos(phi)
        y = r*np.sin(phi)

        if to_plot:
            x = np.append(x, x[0])
            y = np.append(y, y[0])

        return x, y
