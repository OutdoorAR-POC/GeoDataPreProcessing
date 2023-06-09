from pathlib import Path

from topoutils import sphere_sampling
import numpy as np

from topoutils.constants import PROJECT_DIR
from topoutils.obj_reader import ObjFileReader
from topoutils.ply_reader import PlyFileReader
from topoutils.ray_casting import Triangle

model_file_path = PROJECT_DIR.joinpath('models', 'mesh.obj')
model_geometry = ObjFileReader(model_file_path).geometry

N = 8
direction_vectors = sphere_sampling.get_cartesian_coordinates(N)

annotations_directory_path = PROJECT_DIR.joinpath('annotations')
for annotations_file_path in annotations_directory_path.iterdir():
    if annotations_file_path.suffix == '.ply':
        annotations_geometry = PlyFileReader(annotations_file_path).geometry
        print(annotations_geometry.name)
        points = annotations_geometry.vertices
        visibility_maps = np.ones((len(points), N, N)) * np.infty  # each point has its visibility map
        for face in model_geometry.faces:
            triangle = Triangle(*[model_geometry.vertices[vertex_idx] for vertex_idx in face])
            for point_idx, point in enumerate(points):
                intersects, distance = triangle.does_ray_intersect(point, direction_vectors)
                visibility_maps[point_idx] = np.minimum(visibility_maps[point_idx], distance)

        print(visibility_maps)
