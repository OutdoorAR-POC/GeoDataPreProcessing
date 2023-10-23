import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from topoutils.constants import PROJECT_DIR, ASSETS_DIR
from topoutils.obj_reader import ObjFileReader
from topoutils.ply_reader import PlyFileReader
from topoutils.ray_casting import Triangle

cameras_sfm = ASSETS_DIR.joinpath('cameras', 'cameras.sfm')
cameras = json.load(cameras_sfm.open('r'))
intrinsic = cameras['intrinsics'][0]
K = np.array([
    [float(intrinsic["pxFocalLength"]), 0, float(intrinsic["principalPoint"][0]), 0],
    [0, float(intrinsic["pxFocalLength"]), float(intrinsic["principalPoint"][1]), 0],
    [0, 0, 1, 0]
])

views = {view['poseId']: {
    'imgName': view['path'][view['path'].rfind('/') + 1:].upper(),
    'width': int(view['width']),
    'height': int(view['height'])
} for view in cameras['views']}

model_file_path = PROJECT_DIR.joinpath('models', 'decimatedMesh6_closedHoles.obj')
model_geometry = ObjFileReader(model_file_path).geometry

# get all annotated points
annotations_directory_path = PROJECT_DIR.joinpath('annotations')
annotations = np.empty(shape=[0, 3])
annotations_info: list[tuple[str, int]] = []

for annotations_file_path in annotations_directory_path.iterdir():
    if annotations_file_path.suffix == '.ply':
        annotations_geometry = PlyFileReader(annotations_file_path).geometry
        annotations = np.concatenate((annotations, annotations_geometry.vertices))
        num_vertices = len(annotations_geometry.vertices)
        info = zip([annotations_geometry.name] * num_vertices, range(num_vertices))
        annotations_info.extend(info)

images_index = [view['imgName'] for view in views.values()]
results_df = pd.DataFrame(data=None, columns=annotations_info, index=images_index)

for pose_obj in tqdm(cameras['poses']):

    pose = pose_obj['pose']['transform']
    img_name = views[pose_obj['poseId']]['imgName']
    camera_location = np.array([float(x) for x in pose["center"]])
    print(img_name)

    direction_vectors = np.subtract(annotations, camera_location)
    distances = np.array([sum([vi ** 2 for vi in vector]) for vector in direction_vectors])
    z_buffer = np.ones(direction_vectors.shape[:-1]) * np.infty

    for face in model_geometry.faces:
        triangle = Triangle(*[model_geometry.vertices[vertex_idx] for vertex_idx in face])
        intersects, distance = triangle.does_ray_intersect(camera_location, direction_vectors)
        z_buffer = np.minimum(z_buffer, distance)

    results_df.loc[img_name] = (z_buffer > distances).astype(int)

results_df.to_csv(ASSETS_DIR.joinpath("ground_truth.csv"))
