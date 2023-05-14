from unittest import TestCase

import numpy as np

from topoutils import ray_casting, sphere_sampling


class TestRayCasting(TestCase):

    def setUp(self) -> None:
        self.X = [0, 0, 0]
        self.Y = [9, 1, 0]
        self.Z = [7, 8, 0]

    def test_barycentric_coordinates(self):
        P1 = [5, 3, 0]
        expected_coordinates = [0.36923076923076925, 0.3384615384615385, 0.29230769230769227]
        u, v, w = ray_casting.barycentric_coordinates(self.X, self.Y, self.Z, P1)
        self.assertListEqual(expected_coordinates, [u, v, w])
        self.assertEqual(1, u + v + w)

    def test_barycentric_coordinates__when_ray_does_not_intersect_triangle(self):
        P2 = [5, 8, 0]
        expected_coordinates = [0.2153846153846154, 1.0307692307692307, -0.24615384615384606]
        u, v, w = ray_casting.barycentric_coordinates(self.X, self.Y, self.Z, P2)
        self.assertListEqual(expected_coordinates, [u, v, w])
        self.assertEqual(1, u + v + w)

    def test_normal_of_a_triangle(self):
        expected_normal = [0, 0, 65]
        normal = ray_casting.normal_of_a_triangle(self.X, self.Y, self.Z)
        self.assertListEqual(expected_normal, list(normal))

    def test_vector_plane_intersection__when_ray_does_not_intersect_plane(self):
        normal = [0, 0, 65]
        p0 = [5, 3, 6]
        direction_vector = [0, 0, 1]

        _, has_intersection = ray_casting.vector_plane_intersection(
            self.X, normal, p0, direction_vector
        )
        self.assertFalse(has_intersection)

    def test_vector_plane_intersection(self):
        normal = [0, 0, 65]
        p0 = [5, 3, 6]
        direction_vector = [0, 0, -1]

        intersection_point, has_intersection = ray_casting.vector_plane_intersection(
            self.X, normal, p0, direction_vector
        )
        self.assertTrue(has_intersection)
        expected_intersection_point = [0, 0, -6]
        self.assertListEqual(expected_intersection_point, list(intersection_point))

    def test_triangle_init(self):
        triangle = ray_casting.Triangle(self.X, self.Y, self.Z)
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
        triangle = ray_casting.Triangle(self.X, self.Y, self.Z)
        P2 = [5, 8, 0]
        expected_coordinates = [0.2153846153846154, 1.0307692307692307, -0.24615384615384606]
        u, v, w = triangle.barycentric_coordinates(P2)
        self.assertListEqual(expected_coordinates, [u, v, w])
        self.assertEqual(1, u + v + w)

        P1 = [5, 3, 0]
        expected_coordinates = [0.36923076923076925, 0.3384615384615385, 0.29230769230769227]
        u, v, w = triangle.barycentric_coordinates(P1)
        self.assertListEqual(expected_coordinates, [u, v, w])
        self.assertEqual(1, u + v + w)

    def test_triangle_barycentric_coordinates__for_vector_of_points(self):
        triangle = ray_casting.Triangle(self.X, self.Y, self.Z)
        P1 = [5, 3, 0]
        P2 = [5, 8, 0]
        intersection_points = np.array([P1, P2])
        expected_coordinates = np.array([
            [0.36923076923076925, 0.3384615384615385, 0.29230769230769227],
            [0.2153846153846154, 1.0307692307692307, -0.24615384615384606],
        ])
        u, v, w = triangle.barycentric_coordinates(intersection_points)
        barycentric = np.dstack((u, v, w)).squeeze(0)
        np.testing.assert_array_equal(expected_coordinates, barycentric)

    def test_triangle_barycentric_coordinates__for_matrix_of_points(self):
        triangle = ray_casting.Triangle(self.X, self.Y, self.Z)
        P1 = [5, 3, 0]
        P2 = [5, 8, 0]
        intersection_points = np.array([[P1, P2], [P2, P1]])
        coords1 = [0.36923076923076925, 0.3384615384615385, 0.29230769230769227]
        coords2 = [0.2153846153846154, 1.0307692307692307, -0.24615384615384606]
        expected_coordinates = np.array([[coords1, coords2], [coords2, coords1]])
        u, v, w = triangle.barycentric_coordinates(intersection_points)
        barycentric = np.dstack((u, v, w))
        np.testing.assert_array_equal(expected_coordinates, barycentric)

    def test_triangle_does_ray_intersect(self):
        p0 = np.array([5, 3, 6])
        triangle = ray_casting.Triangle(self.X, self.Y, self.Z)

        intersects, distance = triangle.does_ray_intersect(p0, np.array([0, 0, -1]))
        self.assertTrue(intersects)
        self.assertEqual(178, distance)
        self.assertFalse(triangle.does_ray_intersect(p0, np.array([0, 0, 1]))[0])

    def test_triangle_do_multiple_rays_intersect(self):
        # TODO for a matrix
        pass

    def test_vector_plane_intersection_for_multiple_points(self):
        direction_vectors = sphere_sampling.get_cartesian_coordinates(3)
        # direction_vectors is an array of shape (N, N, 3); here N = 3
        X = [0, 0, 0]
        Y = [1, 1, 0]
        Z = [1, 0, 0]
        p0 = [1, 1, 1]
        normal = ray_casting.normal_of_a_triangle(X, Y, Z)
        # let's take the first row
        intersection_point1, has_intersection1 = ray_casting.vector_plane_intersection(
            X, normal, p0, direction_vectors[0, 0, :]
        )
        intersection_point2, has_intersection2 = ray_casting.vector_plane_intersection(
            X, normal, p0, direction_vectors[0, 1, :]
        )
        intersection_point3, has_intersection3 = ray_casting.vector_plane_intersection(
            X, normal, p0, direction_vectors[0, 2, :]
        )
        intersection_points_first_row, has_interesection_first_row = ray_casting.vector_plane_intersection(
            X, normal, p0, direction_vectors[0]
        )
        intersection_points_all_matrix, has_interesection_all_matrix = ray_casting.vector_plane_intersection(
            X, normal, p0, direction_vectors
        )

        self.assertFalse(has_intersection1)
        self.assertFalse(has_intersection2)
        self.assertTrue(has_intersection3)
        np.testing.assert_array_equal(np.array([False, False, True]), has_interesection_first_row)
        np.testing.assert_array_equal(has_interesection_first_row, has_interesection_all_matrix[0])
        np.testing.assert_allclose(intersection_point1, intersection_points_first_row[0])
        np.testing.assert_allclose(intersection_point2, intersection_points_first_row[1])
        np.testing.assert_allclose(intersection_point3, intersection_points_first_row[2])
        np.testing.assert_allclose(intersection_points_all_matrix[0], intersection_points_first_row)
