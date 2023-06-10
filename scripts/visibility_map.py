import json
from dataclasses import asdict

import numpy as np

from topoutils import sphere_sampling
from topoutils.constants import PROJECT_DIR
from topoutils.obj_reader import ObjFileReader
from topoutils.ply_reader import PlyFileReader
from topoutils.ray_casting import Triangle
from topoutils.visibility_writer import Visibility, Vertex, Edge

model_file_path = PROJECT_DIR.joinpath('models', 'mesh.obj')
model_geometry = ObjFileReader(model_file_path).geometry

N = 8
direction_vectors = sphere_sampling.get_cartesian_coordinates(N)

visibility_directory_path = PROJECT_DIR.joinpath('visibility')
annotations_directory_path = PROJECT_DIR.joinpath('annotations')
for annotations_file_path in annotations_directory_path.iterdir():
    if annotations_file_path.suffix == '.ply':
        annotations_geometry = PlyFileReader(annotations_file_path).geometry

        visibility = Visibility(
            name=annotations_geometry.name,
            edges=[Edge(*row.tolist()) for row in annotations_geometry.edges],
        )

        points = annotations_geometry.vertices
        visibility_maps = np.ones((len(points), N, N)) * np.infty  # each point has its visibility map
        for face in model_geometry.faces:
            triangle = Triangle(*[model_geometry.vertices[vertex_idx] for vertex_idx in face])
            for point_idx, point in enumerate(points):
                intersects, distance = triangle.does_ray_intersect(point, direction_vectors)
                visibility_maps[point_idx] = np.minimum(visibility_maps[point_idx], distance)

        for point_idx, point in enumerate(points):
            point_list = point.tolist()
            visibility.vertices.append(
                Vertex(
                    id=point_idx,
                    x=point_list[0],
                    y=point_list[1],
                    z=point_list[2],
                    visibility_grid=visibility_maps[point_idx].ravel(order='F').tolist()
                )
            )

        json.dump(
            asdict(visibility),
            visibility_directory_path.joinpath(annotations_file_path.stem + ".json").open('w'),
            indent=2,
        )

