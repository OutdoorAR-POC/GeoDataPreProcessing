from unittest import TestCase

import numpy as np
import numpy.testing

from topoutils.constants import ANNOTATIONS_DIR
from topoutils.ply_reader import PlyFileReader


class TestPlyFileReader(TestCase):

    def setUp(self) -> None:
        self.file_path = ANNOTATIONS_DIR.joinpath('BluePolyline.ply')

    def test_ply_file_reader(self) -> None:
        reader = PlyFileReader(self.file_path)
        self.assertEqual(4, len(reader.vertices))
        self.assertEqual(4, len(reader.edges))
        geometry = reader.geometry

        expected_vertices = np.array(
            [
                [-0.657208, 1.2409, 2.93151],
                [-0.693995, 1.19847, 3.03879],
                [-0.640229, 1.13182, 2.9818],
                [-0.591398, 1.15814, 2.86143],
            ]
        )
        expected_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

        expected_faces = []

        self.assertEqual("BluePolyline", geometry.name)
        numpy.testing.assert_array_almost_equal(expected_vertices, geometry.vertices)
        numpy.testing.assert_array_almost_equal(expected_edges, geometry.edges)
        numpy.testing.assert_array_almost_equal(expected_faces, geometry.faces)
