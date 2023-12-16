import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from outdoorar.constants import RESOURCES_DIR, MODELS_DIR, CAMERAS_DIR, ANNOTATIONS_DIR
from outdoorar.obj_reader import ObjFileReader
from outdoorar.ply_reader import PlyFileReader
from outdoorar.ray_casting import Triangle
from outdoorar.rendering import get_image_coordinates, is_inside_image

cameras_sfm = CAMERAS_DIR.joinpath('cameras.sfm')
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

model_file_path = MODELS_DIR.joinpath('decimatedMesh_closedHoles.obj')
model_geometry = ObjFileReader(model_file_path).geometry

# get all annotated points
annotations = np.empty(shape=[0, 3])
annotations_info: list[tuple[str, str]] = []

for annotations_file_path in ANNOTATIONS_DIR.iterdir():
    if annotations_file_path.suffix == '.ply':
        annotations_geometry = PlyFileReader(annotations_file_path).geometry
        annotations = np.concatenate((annotations, annotations_geometry.vertices))
        num_vertices = len(annotations_geometry.vertices)
        info = zip([annotations_geometry.name] * num_vertices, range(num_vertices))
        annotations_info.extend(info)

images_index = [view['imgName'] for view in views.values()]
results_df = pd.DataFrame(
    data=None,
    columns=pd.MultiIndex.from_tuples(annotations_info),
    index=images_index,
)
results_df.columns.names = ['Polyline', 'VertexIdx']

for pose_obj in tqdm(cameras['poses']):

    pose = pose_obj['pose']['transform']
    camera_location = np.array([float(x) for x in pose["center"]])

    view = views[pose_obj['poseId']]
    img_name = view['imgName']
    image_width, image_height = view['width'], view['height']

    R = np.array([float(x) for x in pose["rotation"]]).reshape((3, 3), order='F')
    T = - np.matmul(R, np.array(camera_location)[:, np.newaxis])
    M = np.vstack((np.hstack((R, T)), np.array([0, 0, 0, 1])))

    annotations_coordinates = get_image_coordinates(annotations, K, M)
    annotations_visible = is_inside_image(annotations_coordinates, image_width, image_height)

    direction_vectors = np.subtract(annotations, camera_location)
    distances = np.array([sum([vi ** 2 for vi in vector]) for vector in direction_vectors])
    z_buffer = np.ones(direction_vectors.shape[:-1]) * np.infty

    for face in model_geometry.faces:
        triangle = Triangle(*[model_geometry.vertices[vertex_idx] for vertex_idx in face])
        intersects, distance = triangle.does_ray_intersect(camera_location, direction_vectors, 0)
        z_buffer = np.minimum(z_buffer, distance)

    results_df.loc[img_name] = np.logical_and(
        z_buffer > distances,
        annotations_visible,
    ).astype(int)

results_df.to_csv(RESOURCES_DIR.joinpath("ground_truth.csv"))
