from unittest import TestCase

import numpy as np

from topoutils import intersection


class TestIntersection(TestCase):

    def setUp(self) -> None:
        self.X = [0, 0, 0]
        self.Y = [9, 1, 0]
        self.Z = [7, 8, 0]

    def test_barycentric_coordinates(self):
        P1 = [5, 3, 0]
        expected_coordinates = [0.36923076923076925, 0.3384615384615385, 0.29230769230769227]
        u, v, w = intersection.barycentric_coordinates(self.X, self.Y, self.Z, P1)
        self.assertListEqual(expected_coordinates, [u, v, w])
        self.assertEqual(1, u+v+w)

    def test_barycentric_coordinates__when_ray_does_not_intersect_triangle(self):
        P2 = [5, 8, 0]
        expected_coordinates = [0.2153846153846154, 1.0307692307692307, -0.24615384615384606]
        u, v, w = intersection.barycentric_coordinates(self.X, self.Y, self.Z, P2)
        self.assertListEqual(expected_coordinates, [u, v, w])
        self.assertEqual(1, u + v + w)

    def test_normal_of_a_triangle(self):
        expected_normal = [0, 0, 65]
        normal = intersection.normal_of_a_triangle(self.X, self.Y, self.Z)
        self.assertListEqual(expected_normal, list(normal))

    def test_vector_plane_intersection__when_ray_does_not_intersect_plane(self):
        normal = [0, 0, 65]
        p0 = [5, 3, 6]
        direction_vector = [0, 0, 1]

        self.assertIsNone(intersection.vector_plane_intersection(self.X, normal, p0, direction_vector))

    def test_vector_plane_intersection(self):
        normal = [0, 0, 65]
        p0 = [5, 3, 6]
        direction_vector = [0, 0, -1]

        intersection_point = intersection.vector_plane_intersection(self.X, normal, p0, direction_vector)
        expected_intersection_point = [0, 0, -6]
        self.assertListEqual(expected_intersection_point, list(intersection_point))

    def test_triangle_init(self):
        triangle = intersection.Triangle(self.X, self.Y, self.Z)
        expected_normal = [0, 0, 65]
        self.assertListEqual(expected_normal, list(triangle.normal))
        yx = [-9, -1, 0]
        self.assertListEqual(yx, list(triangle.yx))
        yz = [-2, 7, 0]
        self.assertListEqual(yz, list(triangle.yz))

        dot00 = np.dot(yx, yx)
        dot01 = np.dot(yx, yz)
        dot11 = np.dot(yz, yz)
        self.assertEqual(triangle.dot00, dot00)
        self.assertEqual(triangle.dot01, dot01)
        self.assertEqual(triangle.dot11, dot11)
        self.assertEqual(dot00 * dot11 - dot01 * dot01, triangle.barycentric_denom)

    def test_triangle_barycentric_coordinates(self):
        triangle = intersection.Triangle(self.X, self.Y, self.Z)
        P2 = [5, 8, 0]
        expected_coordinates = [0.2153846153846154, 1.0307692307692307, -0.24615384615384606]
        u, v, w = triangle.barycentric_coordinates(P2)
        self.assertListEqual(expected_coordinates, [u, v, w])
        self.assertEqual(1, u + v + w)

        P1 = [5, 3, 0]
        expected_coordinates = [0.36923076923076925, 0.3384615384615385, 0.29230769230769227]
        u, v, w = triangle.barycentric_coordinates(P1)
        self.assertListEqual(expected_coordinates, [u, v, w])
        self.assertEqual(1, u+v+w)

    def test_triangle_does_ray_intersect(self):
        p0 = [5, 3, 6]
        triangle = intersection.Triangle(self.X, self.Y, self.Z)

        self.assertTrue(triangle.does_ray_intersect(p0, [0, 0, -1]))
        self.assertFalse(triangle.does_ray_intersect(p0, [0, 0, 1]))

