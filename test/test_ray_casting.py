from unittest import TestCase

import numpy as np

from outdoorar import ray_casting, sphere_sampling
from outdoorar.constants import MODELS_DIR
from outdoorar.obj_reader import ObjFileReader
from outdoorar.ray_casting import Triangle


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

        intersects, distance = triangle.does_ray_intersect(p0, np.array([[[0, 0, -1]]]), 0)
        self.assertTrue(intersects)
        self.assertEqual(36, distance)
        self.assertFalse(triangle.does_ray_intersect(p0, np.array([[[0, 0, 1]]]))[0])
        self.assertFalse(triangle.does_ray_intersect(p0, np.array([[[0, 1, 0]]]), 0)[0])
        self.assertFalse(triangle.does_ray_intersect(p0, np.array([[[0, -1, 0]]]))[0])

    def test_triangle_do_multiple_rays_intersect(self):
        p0 = np.array([5, 3, 6])
        triangle = ray_casting.Triangle(self.X, self.Y, self.Z)

        intersects, distance = triangle.does_ray_intersect(
            p0, np.array([[[0, 0, -1], [0, 0, 1]], [[0, -1, 0], [0, 1, 0]]]), 0
        )
        expected_intersections = np.array([[True, False], [False, False]])
        np.testing.assert_array_equal(expected_intersections, intersects)
        expected_distances = np.array([[36, np.inf], [np.inf, np.inf]])
        np.testing.assert_array_equal(expected_distances, distance)

    def test_vector_plane_intersection_for_multiple_points(self):
        N = 3
        direction_vectors = sphere_sampling.get_cartesian_coordinates(N**2)
        # direction_vectors is an array of shape (NxN, 3); here N = 3
        X = [0, 0, 0]
        Y = [1, 1, 0]
        Z = [1, 0, 0]
        p0 = [1, 1, 1]
        normal = ray_casting.normal_of_a_triangle(X, Y, Z)
        idx0 = 0
        idx1 = N
        idx2 = 2*N
        # let's take the first row
        intersection_point1, has_intersection1 = ray_casting.vector_plane_intersection(
            X, normal, p0, direction_vectors[idx0, :]
        )
        intersection_point2, has_intersection2 = ray_casting.vector_plane_intersection(
            X, normal, p0, direction_vectors[idx1, :]
        )
        intersection_point3, has_intersection3 = ray_casting.vector_plane_intersection(
            X, normal, p0, direction_vectors[idx2, :]
        )
        intersection_points_all_matrix, has_interesection_all_matrix = ray_casting.vector_plane_intersection(
            X, normal, p0, direction_vectors
        )

        self.assertFalse(has_intersection1)
        self.assertFalse(has_intersection2)
        self.assertTrue(has_intersection3)
        self.assertFalse(has_interesection_all_matrix[idx0])
        self.assertFalse(has_interesection_all_matrix[idx1])
        self.assertTrue(has_interesection_all_matrix[idx2])

        np.testing.assert_allclose(intersection_point1, intersection_points_all_matrix[idx0])
        np.testing.assert_allclose(intersection_point2, intersection_points_all_matrix[idx1])
        np.testing.assert_allclose(intersection_point3, intersection_points_all_matrix[idx2])

    def test_vector_plane_intersection_for_multiple_points__assert_column_major_reshape(self):
        N = 3
        direction_vectors = sphere_sampling.get_cartesian_coordinates_from_spherical(
            *sphere_sampling.get_equal_angle_spherical_coordinates(N**2)
        )
        direction_vectors_flat = sphere_sampling.get_cartesian_coordinates(N**2)
        # direction_vectors is an array of shape (N, N, 3); here N = 3
        X = [0, 0, 0]
        Y = [1, 1, 0]
        Z = [1, 0, 0]
        p0 = [1, 1, 1]
        normal = ray_casting.normal_of_a_triangle(X, Y, Z)

        for i in range(N):
            for j in range(N):
                intersection_point, has_intersection = ray_casting.vector_plane_intersection(
                    X, normal, p0, direction_vectors[i, j, :]
                )
                intersection_point_flat, has_intersection_flat = ray_casting.vector_plane_intersection(
                    X, normal, p0, direction_vectors_flat[j * N + i]
                )
                np.testing.assert_allclose(intersection_point, intersection_point_flat)
                self.assertEqual(has_intersection, has_intersection_flat)

    def test_cube(self):
        triangle = Triangle([0., 0., 0.], [0., 1., 1.], [0., 1., 0.])
        delta = 0.000001
        inside_triangle, squared_distances = triangle.does_ray_intersect(
            np.array([1, 1, 1]),
            np.array([-1, 0, 0]),
            delta,
        )
        self.assertTrue(inside_triangle)
        self.assertAlmostEqual(1, squared_distances, delta=delta * 10)

    def test_cube2(self):
        point = np.array([1/2, 0, 1/2])

        file_path = MODELS_DIR.joinpath('cube.obj')
        geometry = ObjFileReader(file_path).geometry

        N = 4
        direction_vectors = sphere_sampling.get_cartesian_coordinates(N**2)

        triangle = Triangle(*[geometry.vertices[vertex_idx] for vertex_idx in geometry.faces[-1]])
        inside_triangle, squared_distances = triangle.does_ray_intersect(point, direction_vectors)
        self.assertEqual(0, np.sum(inside_triangle))
        self.assertTrue(np.all(np.isinf(squared_distances.ravel())))

        triangle = Triangle(*[geometry.vertices[vertex_idx] for vertex_idx in geometry.faces[9]])
        inside_triangle, squared_distances = triangle.does_ray_intersect(point, direction_vectors)
        self.assertEqual(4, np.sum(inside_triangle))
        self.assertEqual(12, np.sum(np.isinf(squared_distances.ravel())))


        