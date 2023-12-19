from outdoorar.constants import MODELS_DIR
from outdoorar.ground_truth import calculate_visibility_from_full_geometry

small_meshes = [
    MODELS_DIR.joinpath('decimatedMesh_closedHoles_4096.obj'),
    MODELS_DIR.joinpath('decimatedMesh_closedHoles_2048.obj'),
    MODELS_DIR.joinpath('decimatedMesh_closedHoles_1024.obj'),
]

for model_file_path in small_meshes:
    calculate_visibility_from_full_geometry(model_file_path)
