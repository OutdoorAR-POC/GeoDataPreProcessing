from typing import Sequence

import numpy as np


def normal_of_a_triangle(x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return np.cross(y - x, z - x)


def vector_plane_intersection(x, n, p0, v):
    """

    :param x: point on a plane
    :param n: plane normal
    :param p0: starting point of the ray
    :param v: ray vector
    :return: intersection of the ray with the plane, if any
    """

    x = np.array(x)
    n = np.array(n)
    p0 = np.array(p0)
    v = np.array(v)

    denom = np.dot(v, n)
    if np.abs(denom) < 0.000001:
        return None  # ray parallel to the plane

    t = np.dot(x - p0, n) / denom
    if t < 0:
        return None  # the other side of the plane

    return t * v


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
        dot02 = np.dot(self.yx, yp)
        dot12 = np.dot(self.yz, yp)

        # Calculate the barycentric coordinates
        u = (self.dot11 * dot02 - self.dot01 * dot12) / self.barycentric_denom
        v = (self.dot00 * dot12 - self.dot01 * dot02) / self.barycentric_denom
        w = 1.0 - u - v

        return u, v, w

    def does_ray_intersect(self, point, ray_vector) -> tuple[bool, float | None]:
        """This function returns squared distance between the given point and the intersection point of the ray vector
        with the triangle. Square root is taken later, for comparison squared distance is fine, since square function
        preserves monotonicity. """
        intersection_point = vector_plane_intersection(self.x, self.normal, point, ray_vector)
        if intersection_point is None:  # the ray does not intersect the plane
            return False, None
        u, v, w = self.barycentric_coordinates(point)
        if u >= 0 and v >= 0 and w >= 0:
            connector = intersection_point - point
            return True, sum([vi**2 for vi in connector])
        return False, None
