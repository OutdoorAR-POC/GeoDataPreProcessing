import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from outdoorar.constants import RESOURCES_DIR, CAMERAS_DIR, ANNOTATIONS_DIR
from outdoorar.obj_reader import ObjFileReader
from outdoorar.ply_reader import PlyFileReader
from outdoorar.ray_casting import Triangle
from outdoorar.rendering import get_image_coordinates, is_inside_image


def get_cameras(cameras_sfm=CAMERAS_DIR.joinpath('cameras.sfm')):
    return json.load(cameras_sfm.open('r'))


def get_intrinsic_matrix(cameras):
    intrinsic_elements = cameras['intrinsics'][0]
    intrinsic = np.array([
        [
            float(intrinsic_elements["pxFocalLength"]),
            0,
            float(intrinsic_elements["principalPoint"][0]),
            0,
        ],
        [
            0,
            float(intrinsic_elements["pxFocalLength"]),
            float(intrinsic_elements["principalPoint"][1]),
            0,
        ],
        [0, 0, 1, 0]
    ])
    return intrinsic


def get_views(cameras):
    views = {view['poseId']: {
        'imgName': view['path'][view['path'].rfind('/') + 1:].upper(),
        'width': int(view['width']),
        'height': int(view['height'])
    } for view in cameras['views']}
    return views


def get_poses(cameras):
    return cameras['poses']


def get_annotations() -> tuple:
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

    return annotations, annotations_info


def create_results_dataframe(views, annotations_info):
    images_index = [view['imgName'] for view in views.values()]
    results_df = pd.DataFrame(
        data=None, columns=pd.MultiIndex.from_tuples(annotations_info), index=images_index
    )
    results_df.columns.names = ['Polyline', 'VertexIdx']
    return results_df


def get_pose(pose_obj):
    return pose_obj['pose']['transform']


def get_pose_id(pose_obj):
    return pose_obj['poseId']


def get_camera_location(pose):
    return np.array([float(x) for x in pose["center"]])


def get_extrinsic_matrix(pose, camera_location):
    rotation = np.array([float(x) for x in pose["rotation"]]).reshape((3, 3), order='F')
    translation = - np.matmul(rotation, np.array(camera_location)[:, np.newaxis])
    extrinsic = np.vstack((np.hstack((rotation, translation)), np.array([0, 0, 0, 1])))
    return extrinsic


def calculate_z_buffer(direction_vectors, model_geometry, camera_location):
    z_buffer = np.ones(direction_vectors.shape[:-1]) * np.infty

    for face in model_geometry.faces:
        triangle = Triangle(*[model_geometry.vertices[vertex_idx] for vertex_idx in face])
        intersects, distance = triangle.does_ray_intersect(camera_location, direction_vectors, 0)
        z_buffer = np.minimum(z_buffer, distance)

    return z_buffer


def calculate_visibility_from_full_geometry(model_file_path, output_file_name=None):
    if output_file_name is None:
        output_file_name = f"{model_file_path.stem}.csv"

    model_geometry = ObjFileReader(model_file_path).geometry
    cameras = get_cameras()
    views = get_views(cameras)
    intrinsic = get_intrinsic_matrix(cameras)
    annotations, annotations_info = get_annotations()

    results_df = create_results_dataframe(views, annotations_info)

    for pose_obj in tqdm(get_poses(cameras)):
        pose = get_pose(pose_obj)
        camera_location = get_camera_location(pose)

        view = views[get_pose_id(pose_obj)]
        img_name = view['imgName']
        image_width, image_height = view['width'], view['height']

        extrinsic = get_extrinsic_matrix(pose, camera_location)

        annotations_coordinates = get_image_coordinates(annotations, intrinsic, extrinsic)
        annotations_visible = is_inside_image(annotations_coordinates, image_width, image_height)
        direction_vectors = np.subtract(annotations, camera_location)
        distances = np.array([sum([vi ** 2 for vi in vector]) for vector in direction_vectors])

        z_buffer = calculate_z_buffer(direction_vectors, model_geometry, camera_location)
        results_df.loc[img_name] = np.logical_and(
            z_buffer > distances,
            annotations_visible,
        ).astype(int)

    results_df.to_csv(RESOURCES_DIR.joinpath(output_file_name))
