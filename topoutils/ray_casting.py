from typing import Sequence

import numpy as np


def normal_of_a_triangle(x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return np.cross(y - x, z - x)


def vector_plane_intersection(x, n, p0, v) -> tuple[np.ndarray, np.ndarray]:
    """

    :param x: point on a plane
    :param n: plane normal
    :param p0: starting point of the ray
    :param v: ray vector
    :return: intersection of the ray with the plane, if any
    """

    x = np.array(x)
    n = np.array(n)

    t = np.dot(x - p0, n) / np.dot(v, n)

    has_intersection = t >= 0

    if isinstance(t, np.ndarray):
        if len(t.shape) == 1:  # vector
            t = t.reshape((len(t), 1))
        elif len(t.shape) == 2:  # matrix
            v_shape = v.shape
            t = t.reshape((t.shape[0] * t.shape[1], 1))
            v = v.reshape((v.shape[0] * v.shape[1], v.shape[2]))
            return np.multiply(t, v).reshape(v_shape), has_intersection

    return np.multiply(t, v), has_intersection


def barycentric_coordinates(x, y, z, p):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    p = np.array(p)

    # Calculate the vectors from vertex Y to vertices X and Z
    yx = x - y
    yz = z - y

    # Calculate the vector from vertex Y to point P
    yp = p - y

    # Calculate the dot products
    dot00 = np.dot(yx, yx)
    dot01 = np.dot(yx, yz)
    dot02 = np.dot(yx, yp)
    dot11 = np.dot(yz, yz)
    dot12 = np.dot(yz, yp)

    # Calculate the barycentric coordinates
    denom = dot00 * dot11 - dot01 * dot01
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    w = 1.0 - u - v

    return u, v, w


Point = Sequence[float]


class Triangle:
    def __init__(self, x: Point | np.ndarray, y: Point | np.ndarray, z: Point | np.ndarray) -> None:
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.normal = normal_of_a_triangle(x, y, z)
        self.yx = self.x - self.y
        self.yz = self.z - self.y
        self.dot00 = np.dot(self.yx, self.yx)
        self.dot01 = np.dot(self.yx, self.yz)
        self.dot11 = np.dot(self.yz, self.yz)
        self.barycentric_denom = self.dot00 * self.dot11 - self.dot01 * self.dot01

    def barycentric_coordinates(self, p: Point | np.ndarray):
        p = np.array(p)

        # Calculate the vector from vertex Y to point P
        yp = p - self.y

        # Calculate the dot products
        dot02 = np.dot(yp, self.yx)
        dot12 = np.dot(yp, self.yz)

        # Calculate the barycentric coordinates
        u = (self.dot11 * dot02 - self.dot01 * dot12) / self.barycentric_denom
        v = (self.dot00 * dot12 - self.dot01 * dot02) / self.barycentric_denom
        w = 1.0 - u - v

        return u, v, w

    def does_ray_intersect(self, point, ray_vector) -> tuple[bool, float | None | np.ndarray]:
        """This function returns squared distance between the given point and the intersection point of the ray vector
        with the triangle. Square root is taken later, for comparison squared distance is fine, since square function
        preserves monotonicity. """
        squared_distances = np.ones(ray_vector.shape[:-1]) * np.infty
        intersection_point, does_intersect_plane = vector_plane_intersection(self.x, self.normal, point, ray_vector)
        u, v, w = self.barycentric_coordinates(intersection_point)
        inside_triangle = does_intersect_plane & (u >= 0) & (v >= 0) & (w >= 0)
        if len(inside_triangle.shape) == 0 and inside_triangle:
            connector = intersection_point - point
            squared_distances = sum([vi ** 2 for vi in connector])
        else:
            for i, j in zip(*np.where(inside_triangle)):
                connector = intersection_point[i, j] - point
                squared_distances[i, j] = sum([vi**2 for vi in connector])
        return inside_triangle, squared_distances
