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
