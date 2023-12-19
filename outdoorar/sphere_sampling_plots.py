import matplotlib.pyplot as plt
import numpy as np

from outdoorar.constants import FIGURES_DIR
from outdoorar.sphere_sampling import get_equal_angle_spherical_coordinates, get_cartesian_coordinates_from_spherical


def plot_sampling_scheme(n: int) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Make points data
    u, v = get_equal_angle_spherical_coordinates(n)
    coords = get_cartesian_coordinates_from_spherical(u, v, 1)
    x = coords[:, :, 0]
    y = coords[:, :, 1]
    z = coords[:, :, 2]

    ax.plot_surface(x, y, z, color='gainsboro', alpha=1.0)
    ax.scatter(x, y, z, marker='o')

    # Set an equal aspect ratio
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Directional visibility map of a point as a sphere')
    plt.savefig(str(FIGURES_DIR.joinpath('sampling_scheme.png')), bbox_inches='tight')


def plot_sampling_grid(n: int) -> None:
    plt.figure()
    u, v = get_equal_angle_spherical_coordinates(n)
    U, V = np.meshgrid(u, v)
    plt.scatter(U, V)
    plt.title('Directional visibility map of a point as a grid')
    plt.xlabel('Azimuthal angle $\\phi$')
    plt.ylabel('Polar angle $\\theta$')
    plt.xticks([0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2 * np.pi],
               ['0', '$0.5\\pi$', '$\\pi$', '$1.5\\pi$', '$2\\pi$'])
    plt.yticks([0, 0.5 * np.pi, np.pi],
               ['0', '$0.5\\pi$', '$\\pi$'])
    # for
    plt.plot([0, 0], [0, np.pi], color='gainsboro')
    plt.plot([0, np.pi * 2], [0, 0], color='gainsboro')
    for uu in u:
        plt.plot([uu + u[0], uu + u[0]], [0, np.pi], color='gainsboro')
    for vv in v:
        plt.plot([0, 2 * np.pi], [vv + v[0], vv + v[0]], color='gainsboro')
    plt.savefig(str(FIGURES_DIR.joinpath('sampling_grid.png')), bbox_inches='tight')


if __name__ == '__main__':
    plot_sampling_grid(8)
    plot_sampling_scheme(8)
