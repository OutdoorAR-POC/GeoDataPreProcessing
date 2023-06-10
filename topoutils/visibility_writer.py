from dataclasses import dataclass, field


@dataclass
class Vertex:
    id: int
    x: float
    y: float
    z: float
    visibility_grid: list[list[float]] = field(default_factory=list)


@dataclass
class Edge:
    vertex1: int
    vertex2: int


@dataclass
class Visibility:
    name: str
    vertices: list[Vertex] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
