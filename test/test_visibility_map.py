from unittest import TestCase

from topoutils import visibility_map


class TestVisibilityMap(TestCase):

    def test_get_spherical_coordinates(self):
        try:
            visibility_map.get_spherical_coordinates(-1)
            self.fail("Should raise ValueError")
        except ValueError:
            pass
