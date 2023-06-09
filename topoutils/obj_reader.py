from __future__ import annotations

from pathlib import Path

from topoutils.constants import PROJECT_DIR
from topoutils.geometry import Geometry


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
        return Geometry(self._name, self._vertices, faces=self._faces)


if __name__ == '__main__':
    file_path = PROJECT_DIR.joinpath('models', 'cube.obj')
    geometry = ObjFileReader(file_path).geometry
    print(geometry.vertices)
    print(geometry.edges)  # should be empty
    print(geometry.faces)
