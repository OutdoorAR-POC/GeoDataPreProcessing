from outdoorar.constants import MODELS_DIR
from outdoorar.ground_truth import calculate_visibility_from_full_geometry

model_file_path = MODELS_DIR.joinpath('decimatedMesh_closedHoles.obj')
calculate_visibility_from_full_geometry(model_file_path, "ground_truth.csv")
