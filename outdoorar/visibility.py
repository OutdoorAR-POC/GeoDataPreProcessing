import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from dacite import from_dict

from outdoorar.sphere_sampling import get_equal_angle_spherical_coordinates


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


def calculate_visibility(vertices: list[Vertex], eye: list[float]) -> np.ndarray:
    n = int(np.sqrt(len(vertices[0].visibility_grid)))
    u, v = get_equal_angle_spherical_coordinates(n)
    points = np.array([[v.x, v.y, v.z] for v in vertices])
    delta = eye - points
    R = np.sqrt(np.sum(np.square(delta), axis=1))
    polar_angle = (np.arccos(delta[:, 2] / R) % np.pi).reshape(-1, 1)
    azimuthal_angle = (np.arctan2(delta[:, 1], delta[:, 0]) % (2 * np.pi)).reshape(-1, 1)
    azimuthal_idx = np.argmin(np.abs(u - azimuthal_angle), axis=1)
    polar_idx = np.argmin(np.abs(v - polar_angle), axis=1)
    poly_vis_idx = polar_idx * n + azimuthal_idx
    nn_visibility = [vertex.visibility_grid[vis_idx] for vis_idx, vertex in zip(poly_vis_idx, vertices)]
    return nn_visibility >= R
