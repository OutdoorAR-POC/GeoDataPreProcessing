from pathlib import Path

from topoutils import sphere_sampling
import numpy as np

from topoutils.obj_reader import ObjFileReader
from topoutils.ray_casting import Triangle

file_path = Path(__file__).parents[1].joinpath('models', 'mesh.obj')
geometry = ObjFileReader(file_path).geometry
points = np.array([
    [1, 1, 1],  # corner
    # [1 / 2, 0, 1 / 2],  # center of a face
    # [1 / 2, 1 / 2, 1 / 2],  # inside - może to nie ma sensu?
    # [-1, 1 / 2, 2],  # outside, diagonal to the middle of an edge
])
N = 8
direction_vectors = sphere_sampling.get_cartesian_coordinates(N)
visibility_maps = np.ones((len(points), N, N)) * np.infty  # each point has its visibility map
for face in geometry.faces:
    triangle = Triangle(*[geometry.vertices[vertex_idx] for vertex_idx in face])
    for point_idx, point in enumerate(points):
        intersects, distance = triangle.does_ray_intersect(point, direction_vectors)
        visibility_maps[point_idx] = np.minimum(visibility_maps[point_idx], distance)

print(visibility_maps)
