import numpy as np


def get_image_coordinates(polyline: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Transforms points in world coordinates into image coordinates.

    :param polyline: an `n x 3` matrix of points in world coordinates
    :param intrinsic: camera intrinsic matrix 3 x 4
    :param extrinsic: camera extrinsic matrix 4 x 4
    :return: a `3 x n` matrix of points in image coordinates
    """
    polyline = np.hstack((polyline[:, :3], np.ones((polyline.shape[0], 1))))
    points = np.matmul(intrinsic, np.matmul(extrinsic, polyline.T))
    return np.divide(points, points[-1, :]).astype(int)


def is_inside_image(curr_image_coords: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    """Checks whether point coordinates are inside an image.

    :param curr_image_coords: homogenous coordinates of points in an image, matrix `3 x n`, where `n` is the number
    of points.
    :param image_width: image width
    :param image_height: image height
    :return: a boolean ndarray of size `1 x n`, where `n` is the number of points.
    """
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
