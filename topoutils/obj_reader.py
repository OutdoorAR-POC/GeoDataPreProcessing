from __future__ import annotations

from pathlib import Path

import numpy as np


class Geometry:
    def __init__(self, name: str, vertices: list, faces: list) -> None:
        self._vertices = np.array(vertices)
        self._faces = np.array(faces)
        self._name = name or ''

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def vertices(self) -> np.array:
        return self._vertices

    @property
    def faces(self) -> np.array:
        return self._faces


class ObjFileReader:

    def __init__(self, obj_file_path: Path):
        self._name = None
        self._vertices = []
        self._faces = []
        with obj_file_path.open('rt') as input_file:
            for line in input_file:
                self.parse_line(line)

    def parse_line(self, line: str | None) -> None:
        tokens = line.rstrip('\n').split()
        if len(tokens) == 0:
            return
        match tokens[0]:
            case 'g':
                if len(tokens) > 1:
                    self._name = tokens[1]
            case 'v':
                self._vertices.append([float(x) for x in tokens[1:]])
            case 'f':
                self._faces.append([
                    int(token.split('/')[0])-1  # 0-indexing
                    for token in tokens[1:]
                ])

    @property
    def geometry(self) -> Geometry:
        return Geometry(self._name, self._vertices, self._faces)


if __name__ == '__main__':
    file_path = Path(__file__).parents[1].joinpath('models', 'cube.obj')
    geometry = ObjFileReader(file_path).geometry
