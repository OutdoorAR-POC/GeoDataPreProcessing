from unittest import TestCase

import numpy as np

from topoutils import sphere_sampling


class TestVisibilityMap(TestCase):

    def test_get_spherical_coordinates(self):
        try:
            sphere_sampling.get_spherical_coordinates(-1)
            self.fail("Should raise ValueError")
        except ValueError:
            pass

    def test_get_cartesian_coordinates_from_spherical(self):
        u = v = [0, np.pi / 6, np.pi/4, np.pi/3, np.pi/2]
        coords = sphere_sampling.get_cartesian_coordinates_from_spherical(u, v)
        self.assertEqual((5, 5, 3), coords.shape)
        i = 2  # u[i] = np.pi / 4 = 45 deg, cos = sqrt(2)/2,  sin=sqrt(2)/2
        j = 3  # v[j] = np.pi / 3 = 60 deg, cos = 0.5, sin = sqrt(3)/2
        expected_result = np.array([
            np.sqrt(2)/2 * np.sqrt(3) / 2,  # cos(u) * sin(v)
            np.sqrt(2) / 2 * np.sqrt(3)/2,  # sin(u) * sin(v)
            0.5  # cos(v)
        ])
        np.testing.assert_allclose(expected_result, coords[i, j, :])
