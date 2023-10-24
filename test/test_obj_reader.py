from pathlib import Path
from unittest import TestCase

from outdoorar.constants import MODELS_DIR
from outdoorar.obj_reader import ObjFileReader


class TestObjReader(TestCase):

    def setUp(self) -> None:
        self.file_path = MODELS_DIR.joinpath('cube.obj')

    def test_obj_file_reader(self) -> None:
        geometry = ObjFileReader(self.file_path).geometry
        self.assertEqual(8, len(geometry.vertices))
        self.assertEqual("cube", geometry.name)
        self.assertEqual(12, len(geometry.faces))
        self.assertListEqual([4, 6, 7], geometry.faces[6].tolist())
