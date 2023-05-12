from pathlib import Path
from unittest import TestCase

import numpy as np

from topoutils.obj_reader import ObjFileReader


class TestObjReader(TestCase):

    def setUp(self) -> None:
        self.file_path = Path(__file__).parents[1].joinpath('models', 'cube.obj')

    def test_obj_file_reader(self) -> None:
        geometry = ObjFileReader(self.file_path).geometry
        self.assertEqual(8, len(geometry.vertices))
        self.assertEqual("cube", geometry.name)
        self.assertEqual(12, len(geometry.faces))
        self.assertListEqual([4, 6, 7], geometry.faces[6])
