from unittest import TestCase

import numpy as np
import numpy.testing as npt

from outdoorar import visibility
from outdoorar.sphere_sampling import SamplingScheme
from outdoorar.visibility import Vertex


class TestVisibilityCalculation(TestCase):

    def setUp(self) -> None:
        self.vertices = [
            Vertex(id=0, x=2.09792, y=0.0161036, z=2.19078,
                   visibility_grid=[np.inf, 0.0024356897697369803, 0.008761405794143395,
                                    0.006770713251874363]),
            Vertex(id=1, x=1.97127, y=0.292286, z=2.52505,
                   visibility_grid=[np.inf, 0.2125032148812034, 0.00012423673679289537,
                                    0.00033479276213234117]),
            Vertex(id=2, x=1.83196, y=0.704518, z=2.65416,
                   visibility_grid=[np.inf, 0.41752532945352633, 1.3234047751308612e-11,
                                    7.656296401590816e-12]),
            Vertex(id=3, x=1.78961, y=1.11065, z=2.69233,
                   visibility_grid=[np.inf, 2.067983465570001, 9.087285737208367e-06,
                                    1.0628462363310542e-05]),
            Vertex(id=4, x=1.54526, y=1.34361, z=2.79223,
                   visibility_grid=[np.inf, np.inf, 0.00017461204072327283, 7.514521634656027e-05]),
            Vertex(id=5, x=1.41968, y=2.06772, z=2.8332,
                   visibility_grid=[np.inf, np.inf, np.inf, 8.504845253698063e-05]),
        ]
        self.eye = [6.08202209269405, 1.4887606714272859, 1.124454019938587]
        self.points = visibility.vertices_to_points(self.vertices)
        self.samples = 4

    def test_calculate_visibility_index(self):
        points_to_camera_vectors = self.eye - self.points
        points_to_camera_distances = np.sqrt(np.sum(np.square(points_to_camera_vectors), axis=1))

        visibility1 = visibility.get_visibility_index_by_cosine_distance(
            points_to_camera_vectors, self.samples, SamplingScheme.EQUAL_ANGLE
        )
        visibility2 = visibility.get_visibility_index_equal_sampling(
            points_to_camera_vectors, points_to_camera_distances, self.samples
        )
        npt.assert_array_almost_equal(visibility1, visibility2)

    def test_calculate_visibility(self):
        visibility1 = visibility.calculate_visibility(
            self.vertices,
            self.eye,
            SamplingScheme.EQUAL_ANGLE,
            visibility.NearestNeighborSelector.EQUAL_SPACING,
        )
        visibility2 = visibility.calculate_visibility(
            self.vertices,
            self.eye,
            SamplingScheme.EQUAL_ANGLE,
            visibility.NearestNeighborSelector.COSINE_DISTANCE,
        )
        npt.assert_array_almost_equal(visibility1, visibility2)
