from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import json
from dacite import from_dict


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
