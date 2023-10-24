import numpy as np


def get_image_coordinates(polyline: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    polyline = np.hstack((polyline[:, :3], np.ones((polyline.shape[0], 1))))
    points = np.matmul(intrinsic, np.matmul(extrinsic, polyline.T))
    return np.divide(points, points[-1, :]).astype(int)


def is_inside_image(curr_image_coords: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    return np.logical_and(
        np.logical_and(
            curr_image_coords[0] >= 0,
            curr_image_coords[0] <= image_width,
        ),
        np.logical_and(
            curr_image_coords[1] >= 0,
            curr_image_coords[1] <= image_height
        )
    )
