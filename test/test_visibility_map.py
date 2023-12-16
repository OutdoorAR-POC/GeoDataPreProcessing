# Let the cube be a geometry and let there be two points:
# Point A: vertex of a cube
# Point B: centre of the cube's face
# Calculate visibility maps for these two points.

from unittest import TestCase

import numpy as np
import numpy.testing as npt

from outdoorar import sphere_sampling
from outdoorar.constants import MODELS_DIR
from outdoorar.obj_reader import ObjFileReader
from outdoorar.ray_casting import Triangle


class TestVisibilityMap(TestCase):

    def setUp(self) -> None:
        file_path = MODELS_DIR.joinpath('cube.obj')
        self.geometry = ObjFileReader(file_path).geometry
        self.points = np.array([
            [1, 1, 1],  # corner
            [1 / 2, 0, 1 / 2],  # center of a face
            [1 / 2, 1 / 2, 1 / 2],  # inside
            [-1, 1 / 2, 2],  # outside, diagonal to the middle of an edge
        ])
        self.N = 8
        self.direction_vectors = sphere_sampling.get_cartesian_coordinates(self.N**2)

    def test_visibility_map(self):
        direction_vectors2 = sphere_sampling.get_cartesian_coordinates_from_spherical(
            *sphere_sampling.get_equal_angle_spherical_coordinates(self.N**2)
        )
        # assume infinite visibility
        # each point has its visibility map
        visibility_maps = np.ones((len(self.points), self.N * self.N)) * np.infty
        visibility_maps2 = np.ones((len(self.points), self.N, self.N)) * np.infty

        for face in self.geometry.faces:
            triangle = Triangle(*[self.geometry.vertices[vertex_idx] for vertex_idx in face])
            for point_idx, point in enumerate(self.points):
                intersects, distance = triangle.does_ray_intersect(point, self.direction_vectors)
                visibility_maps[point_idx] = np.min((visibility_maps[point_idx], distance))

                intersects2, distance2 = triangle.does_ray_intersect(point, direction_vectors2)
                visibility_maps2[point_idx] = np.min((visibility_maps2[point_idx], distance2))

        visibility_maps2 = visibility_maps2.reshape(visibility_maps2.shape[:-2]+(-1,), order='F')

        npt.assert_array_almost_equal(visibility_maps, visibility_maps2)
