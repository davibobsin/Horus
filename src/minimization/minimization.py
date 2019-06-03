"""This script derives the most probable profile of a
regular prism from the measurements of a camera"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from camera import Camera
from section import Section

# https://stats.stackexchange.com/questions/138325/clustering-a-correlation-matrix


def func(theta, r, phi, h):
    """func"""
    psy = np.add.outer(phi, theta)
    return r*np.cos(psy)/(h - r*np.sin(psy))


def jacobian(theta, r, phi, h):
    """jacobian"""
    psy = phi + theta
    sin = np.sin(psy)
    cos = np.cos(psy)
    relative_height = h - r*sin
    partial_derivative_radius = (
        cos*relative_height-r*sin*cos)/(relative_height**2)
    partial_derivative_phi = (
        (r*cos)**2-r*sin*relative_height)/(relative_height**2)
    return np.column_stack((partial_derivative_radius, partial_derivative_phi))


def guess(theta, alpha, h):
    """guess"""
    i_max = alpha.argmax()
    alpha_max = alpha[i_max]
    return np.array([h*np.sin(alpha_max), alpha_max-theta[i_max]])


def normpdf(x, mean, std):
    """normpdf"""
    denom = std*(2*np.pi)**.5
    num = np.exp(-(x-mean)**2/(2*std**2))
    return num/denom


def std(x, mean):
    """std"""
    err = (x-mean)
    var = np.dot(err, err)
    return var**.5


def run(theta, alpha, func, jac, guess, plt_local, kernel=5, noiseless=np.array([])):
    """run"""
    half_kernel = kernel//2

    trimmed_theta = theta[half_kernel:-half_kernel]
    trimmed_alpha = alpha[half_kernel:-half_kernel]

    error_list = []

    for i in range(half_kernel, theta.size - half_kernel):
        kernel_theta = theta[i - half_kernel:i + half_kernel]
        kernel_alpha = alpha[i - half_kernel:i + half_kernel]
        first_guess = guess(kernel_theta, kernel_alpha)
        params, params_covariance = optimize.curve_fit(
            func, kernel_theta, kernel_alpha, p0=first_guess, jac=jac)

        r = params[0]
        phi = params[1] % (2*np.pi)

        fitted = func(trimmed_theta, r, phi)

        error = trimmed_alpha-fitted

        error_list.append(error)

    error_2d = np.stack(error_list, axis=0)

    error_reciprocal = error_2d+error_2d.T

    correlation = normpdf(error_reciprocal, 0, std(error_reciprocal, 0)/100)

    correlation = (correlation.T+correlation)/2
    correlation = (correlation-correlation.min()) / \
        (correlation.max()-correlation.min())

    dissimilarity = 1-correlation
    np.fill_diagonal(dissimilarity, 0)
    dissimilarity = (dissimilarity.T+dissimilarity)/2

    hierarchy = linkage(squareform(dissimilarity), method='average')
    labels = fcluster(hierarchy, 0.5, criterion='distance')

    unique, indices, unique_counts = np.unique(
        labels, return_counts=True, return_inverse=True)

    filtered_labels = unique[indices]

    # u = l2u(filtered_labels)

    fitted_list = []
    params_list = []

    for x in unique[unique_counts > kernel]:
        segment_indices = (labels == x)

        local_error = dissimilarity[:, segment_indices][segment_indices, :]
        vertical_error = dissimilarity[:, segment_indices]

        cluster_error = sum(local_error, 0)/local_error.shape[0]
        cluster_alpha = trimmed_alpha[segment_indices]
        cluster_theta = trimmed_theta[segment_indices]

        plt.plot(np.degrees(cluster_theta), cluster_error)

        first_guess = guess(cluster_theta, cluster_alpha)

        params, params_covariance = optimize.curve_fit(
            func, cluster_theta, cluster_alpha, p0=first_guess, jac=jac, sigma=cluster_error)

        r = params[0]
        phi = params[1] % (2*np.pi)

        fitted = func(trimmed_theta, r, phi)

        fitted_list.append(fitted)
        params_list.append([r, phi])

    fitted_2d = np.stack(fitted_list, axis=0)

    params_list = sorted(params_list, key=lambda l: l[1], reverse=True)

    r = params_list[:][0]
    phi = params_list[:][1]

    Xs, Ys = Section.to_xy(r, phi, True)

    if PLOT:
        # plt_local.figure(figsize=(6, 4))
        # plt_local.imshow(correlation)

        plt_local.figure(figsize=(6, 4))
        # plt_local.imshow(u)

        plt_local.figure(figsize=(6, 4))
        plt_local.imshow(dissimilarity)

        # plt_local.figure(figsize=(6, 4))
        # plt_local.imshow(np.degrees(fitted_2d))

        plt_local.figure(figsize=(6, 4))
        plt_local.plot(Xs, Ys, marker='o', color=COLORS[2])
        plt.gca().set_aspect('equal', adjustable='box')

        plt_local.figure(figsize=(6, 4))
        plt.subplot(2, 1, 2)
        plt_local.scatter(np.degrees(trimmed_theta),
                          np.degrees(trimmed_alpha), color=COLORS[1])
        plt_local.plot(np.degrees(trimmed_theta), np.degrees(
            fitted_2d.max(0)), color=COLORS[2])
        plt_local.scatter(np.degrees(trimmed_theta),
                          filtered_labels, color=COLORS[3])
        if noiseless.any():
            plt_local.plot(np.degrees(theta), np.degrees(
                noiseless), color=COLORS[0])
        plt.subplot(2, 1, 1)
        plt_local.plot(Xs, Ys, marker='o', color=COLORS[2])
        plt.gca().set_aspect('equal', adjustable='box')

    return r, phi


def main():
    """main"""

    h = 15

    def f(theta, r, phi):
        return func(theta, r, phi, h)

    def j(theta, r, phi):
        return jacobian(theta, r, phi, h)

    def g(theta, alpha):
        return guess(theta, alpha, h)

    points = np.array([[1, 0], [0, 1], [-1, 0], [-1, -1], [1, -1]])
    # points = np.array([[1,0.5],[0,1],[-1,0.5],[-1,-.5],[0,-1],[1,-.5]])
    # points = np.array([[1,0],[0,1],[-1,0],[0,-1]])
    section = Section(points)

    length = 36*2
    theta = np.linspace(0, 2*np.pi, length)

    result = section.rotate(theta)

    cam = Camera(h)

    # Seed the random number generator for reproducibility
    np.random.seed(0)

    true_alpha = cam.measure(result)
    noisy_alpha = true_alpha + np.random.normal(size=length)/2000

    r = h*np.sin(noisy_alpha)
    phi = noisy_alpha-theta

    Xs, Ys = section.to_xy(r, phi, True)

    run(theta, noisy_alpha, func=f, jac=j, guess=g,
        kernel=5, plt_local=plt, noiseless=true_alpha)

    if PLOT:
        section.plot(plt)
        plt.plot(Xs, Ys, marker='o', color=COLORS[1])
        plt.gca().set_aspect('equal', adjustable='box')

    #     plt.figure(figsize=(6, 4))
    #     plt.subplot(2, 1, 1)
    #     section.Plot(plt)
    #     plt.subplot(2, 1, 2)
    #     plt.plot(np.degrees(theta), np.degrees(true_alpha))

        plt.figure(figsize=(6, 4))
        plt.subplot(2, 1, 1)
        section.plot(plt)
        plt.plot(Xs, Ys, marker='o', color=COLORS[1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.subplot(2, 1, 2)
        plt.plot(np.degrees(theta), np.degrees(true_alpha))
        plt.scatter(np.degrees(theta), np.degrees(
            noisy_alpha), color=COLORS[1])

    plt.show()


PLOT = False
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
main()
