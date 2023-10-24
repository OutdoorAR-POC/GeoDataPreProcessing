import json
from dataclasses import asdict

import numpy as np

from outdoorar import sphere_sampling
from outdoorar.constants import MODELS_DIR, VISIBILITY_DIR, ANNOTATIONS_DIR
from outdoorar.obj_reader import ObjFileReader
from outdoorar.ply_reader import PlyFileReader
from outdoorar.ray_casting import Triangle
from outdoorar.visibility import Visibility, Vertex, Edge

model_file_path = MODELS_DIR.joinpath('decimatedMesh_closedHoles.obj')
model_geometry = ObjFileReader(model_file_path).geometry

n_range = [2, 4, 8, 16, 32]
for N in n_range:
    direction_vectors = sphere_sampling.get_cartesian_coordinates(N)

    visibility_directory_path = VISIBILITY_DIR.joinpath(f'n_{N}')
    visibility_directory_path.mkdir(exist_ok=True, parents=True)

    for annotations_file_path in ANNOTATIONS_DIR.iterdir():
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
                    intersects, distance = triangle.does_ray_intersect(point, direction_vectors, 0)
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

