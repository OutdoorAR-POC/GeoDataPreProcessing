from enum import Enum

import numpy as np


class SamplingScheme(Enum):
    EQUAL_ANGLE = 1
    GOLDEN_SPIRAL = 2


def get_golden_spiral_cartesian_coordinates(samples: int) -> np.ndarray:
    if samples <= 0:
        raise ValueError("Number of spherical coordinates must be positive")
    points = np.zeros((samples, 3))
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points[i, 0] = x
        points[i, 1] = y
        points[i, 2] = z

    return points


def get_equal_angle_spherical_coordinates(samples: int) -> tuple:
    if samples <= 0:
        raise ValueError("Number of spherical coordinates must be positive")
    n = int(np.sqrt(samples))
    if n*n != samples:
        raise ValueError(f"Number of samples must be the square of an integer {samples} != {n}**2")

    u_offset = 2 * np.pi / (2 * n)
    v_offset = np.pi / (2 * n)
    u = np.linspace(u_offset, 2 * np.pi - u_offset, n)
    v = np.linspace(v_offset, np.pi - v_offset, n)
    return u, v


def get_cartesian_coordinates_from_spherical(u, v, r=1) -> np.ndarray:
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    return np.dstack((x, y, z))


def get_cartesian_coordinates(
        n: int, sampling_scheme: SamplingScheme = SamplingScheme.EQUAL_ANGLE,
) -> np.ndarray:
    match sampling_scheme:
        case SamplingScheme.EQUAL_ANGLE:
            return get_cartesian_coordinates_from_spherical(
                *get_equal_angle_spherical_coordinates(n)
            ).reshape((n, 3), order='F')
        case SamplingScheme.GOLDEN_SPIRAL:
            return get_golden_spiral_cartesian_coordinates(n)

