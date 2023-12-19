import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List

import numpy as np
from dacite import from_dict

from outdoorar import sphere_sampling
from outdoorar.sphere_sampling import SamplingScheme


class NearestNeighborSelector(Enum):
    EQUAL_SPACING = 1
    COSINE_DISTANCE = 2


@dataclass
class Vertex:
    id: int
    x: float
    y: float
    z: float
    visibility_grid: List[float] = field(default_factory=list)


@dataclass
class Edge:
    vertex1: int
    vertex2: int


@dataclass
class Visibility:
    name: str
    vertices: List[Vertex] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)


def from_json(path_to_file: Path) -> Visibility:
    return from_dict(data_class=Visibility, data=json.load(path_to_file.open('r')))


def get_visibility_index(
    points_to_camera_vectors: np.ndarray,
    points_to_camera_distances: np.ndarray,
    samples: int,
    sampling_scheme: SamplingScheme,
    algorithm: NearestNeighborSelector,
) -> np.ndarray:
    match algorithm:
        case NearestNeighborSelector.EQUAL_SPACING:
            return get_visibility_index_equal_sampling(
                points_to_camera_vectors,
                points_to_camera_distances,
                samples,
            )
        case NearestNeighborSelector.COSINE_DISTANCE:
            return get_visibility_index_by_cosine_distance(
                points_to_camera_vectors,
                samples,
                sampling_scheme,
            )


def get_visibility_index_by_cosine_distance(
    points_to_camera_vectors: np.ndarray,
    samples: int,
    sampling_scheme: SamplingScheme,
) -> np.ndarray:
    direction_vectors = sphere_sampling.get_cartesian_coordinates(samples, sampling_scheme)
    return np.argmax(
        np.dot(points_to_camera_vectors, direction_vectors.transpose()), axis=1
    )[:, np.newaxis]


def get_visibility_index_equal_sampling(
    points_to_camera_vectors: np.ndarray,
    points_to_camera_distances: np.ndarray,
    samples: int,
) -> np.ndarray:
    sqrt_n = int(np.sqrt(samples))
    polar_angle = (
            np.arccos(points_to_camera_vectors[:, 2] / points_to_camera_distances) % np.pi
    ).reshape(-1, 1)
    azimuthal_angle = (
            np.arctan2(points_to_camera_vectors[:, 1], points_to_camera_vectors[:, 0]) % (2 * np.pi)
    ).reshape(-1, 1)
    azimuthal_delta = 2 * np.pi / sqrt_n
    polar_delta = np.pi / sqrt_n
    azimuthal_idx = azimuthal_angle // azimuthal_delta
    polar_idx = polar_angle // polar_delta
    poly_vis_idx = polar_idx * sqrt_n + azimuthal_idx
    return poly_vis_idx.astype(int)


def calculate_visibility(
    vertices: list[Vertex],
    eye: list[float],
    sampling_scheme: SamplingScheme,
    algorithm: NearestNeighborSelector,
) -> np.ndarray:
    points = vertices_to_points(vertices)
    points_to_camera_vectors = eye - points
    points_to_camera_distances = np.sqrt(np.sum(np.square(points_to_camera_vectors), axis=1))
    poly_vis_idx = get_visibility_index(
        points_to_camera_vectors,
        points_to_camera_distances,
        len(vertices[0].visibility_grid),
        sampling_scheme,
        algorithm,
    )
    nn_visibility = [
        vertex.visibility_grid[vis_idx] for vis_idx, vertex in zip(poly_vis_idx.ravel(), vertices)
    ]
    return nn_visibility >= points_to_camera_distances


def vertices_to_points(vertices: list[Vertex]) -> np.ndarray:
    return np.array([[v.x, v.y, v.z] for v in vertices])
