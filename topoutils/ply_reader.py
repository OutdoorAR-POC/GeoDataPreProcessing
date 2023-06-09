from pathlib import Path
from typing import Callable

from topoutils.constants import PROJECT_DIR
from topoutils.geometry import Geometry


class PlyFileReader:
    def __init__(self, obj_file_path: Path):
        self._name = obj_file_path.stem
        self._vertex_properties = []
        self._vertex_types = []
        self._vertices = []
        self._edge_properties = []
        self._edge_types = []
        self._edges = []
        self._vertex_properties_processing = False
        self._face_properties_processing = False
        self._num_vertices = 0
        self._num_faces = 0
        self._header_complete = False
        with obj_file_path.open('rt') as input_file:
            for line in input_file:
                if not self._header_complete:
                    self.parse_header(line)
                elif len(self._vertices) < self._num_vertices:
                    self.parse_vertex(line)
                elif len(self._edges) < self._num_faces:
                    self.parse_edge(line)
                else:
                    raise ValueError(f"Unexpected line {line}")

    @property
    def edges(self):
        return self._edges

    @property
    def vertices(self):
        return self._vertices

    def parse_header(self, line: str | None) -> None:
        tokens = line.rstrip('\n').split()
        if len(tokens) == 0:
            return

        match tokens[0]:
            case 'ply':
                return
            case 'format':
                if tokens[1] != 'ascii':
                    raise ValueError(f"Unknown file format {line}")
                return
            case 'element':
                match tokens[1]:
                    case 'vertex':
                        self._vertex_properties_processing = True
                        self._face_properties_processing = False
                        self._num_vertices = int(tokens[2])
                    case 'edge':
                        self._vertex_properties_processing = False
                        self._face_properties_processing = True
                        self._num_faces = int(tokens[2])
                    case _:
                        self._vertex_properties_processing = False
                        self._face_properties_processing = False
            case 'property':
                if self._vertex_properties_processing:
                    self._vertex_types.append(self._get_python_type(tokens[1]))
                    self._vertex_properties.append(tokens[2])
                elif self._face_properties_processing:
                    self._edge_types.append(self._get_python_type(tokens[1]))
                    self._edge_properties.append(tokens[2])
            case 'end_header':
                self._vertex_properties_processing = False
                self._face_properties_processing = False
                self._header_complete = True
            case _:
                raise ValueError(f"Unexpected token {tokens[0]}")

    @classmethod
    def _get_python_type(cls, ply_type: str) -> Callable:
        match ply_type:
            case 'uchar' | 'char' | 'ushort' | 'short' | 'uint' | 'int':
                return int
            case 'float' | 'double':
                return float
            case _:
                raise ValueError('Unknown type')

    def parse_vertex(self, line: str | None) -> None:
        tokens = line.rstrip('\n').split()
        if len(tokens) == 0:
            return  # empty line, perhaps?

        self._vertices.append(
            dict(
                zip(
                    self._vertex_properties,
                    [func(x) for x, func in zip(tokens, self._vertex_types)]
                )
            )
        )

    def parse_edge(self, line: str | None) -> None:
        # self._faces.append([int(token.split('/')[0])-1 for token in tokens[1:]])
        tokens = line.rstrip('\n').split()
        if len(tokens) == 0:
            return  # empty line, perhaps?

        self._edges.append(
            dict(
                zip(
                    self._edge_properties,
                    [func(x) for x, func in zip(tokens, self._edge_types)]
                )
            )
        )

    @property
    def geometry(self) -> Geometry:
        vertices = [[x['x'], x['y'], x['z']] for x in self._vertices]
        edges = [[x['vertex1'], x['vertex2']] for x in self._edges]
        return Geometry(self._name, vertices, edges=edges)


if __name__ == '__main__':
    file_path = PROJECT_DIR.joinpath('annotations', 'BluePolyline.ply')
    reader = PlyFileReader(file_path)
    print(reader.vertices)
    print(reader.edges)
    geometry = reader.geometry
    print(geometry.vertices)
    print(geometry.edges)
    print(geometry.faces)  # should be empty
