import numpy as np


def normal_of_a_triangle(x, y, z):
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    return np.cross(y-x, z-x)


def plane_intersection(x, n, p0, v):
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
    v0 = x - y
    v1 = z - y

    # Calculate the vector from vertex Y to point P
    v2 = p - y

    # Calculate the dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    # Calculate the barycentric coordinates
    denom = dot00 * dot11 - dot01 * dot01
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    w = 1.0 - u - v

    return u, v, w

if __name__ == '__main__':

    p0 = [5, 3, 6]
    v1 = [0, 0, 1]
    v2 = [0, 0, -1]
    X = [0, 0, 0]
    Y = [9, 1, 0]
    Z = [7, 8, 0]
    P1 = [5, 3, 0]
    P2 = [5, 8, 0]

    u, v, w = barycentric_coordinates(X, Y, Z, P1)
    print(u, v, w)
    print(u+v+w)
    u, v, w = barycentric_coordinates(X, Y, Z, P2)
    print(u, v, w)
    print(u+v+w)

    n = normal_of_a_triangle(X, Y, Z)
    _p1 = plane_intersection(X, n, p0, v1)
    if _p1 is not None:
        u, v, w = barycentric_coordinates(X, Y, Z, _p1)
        print(u, v, w)
        print(u + v + w)
    else:
        print(f'no intersection for plane and vector {v1}')
    _p2 = plane_intersection(X, n, p0, v2)
    if _p2 is not None:
        u, v, w = barycentric_coordinates(X, Y, Z, _p2)
        print(u, v, w)
        print(u + v + w)
    else:
        print(f'no intersection for plane and vector {v2}')


