from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

obj_file_path = Path(__file__).parents[1].joinpath("models", "cube.obj")


def get_spherical_coordinates(n: int) -> tuple:
    if n <= 0:
        raise ValueError("Number of spherical coordinates must be positive")
    u_offset = 2 * np.pi / (2 * n)
    v_offset = np.pi / (2 * n)
    u = np.linspace(u_offset, 2 * np.pi - u_offset, n)
    v = np.linspace(v_offset, np.pi - v_offset, n)
    return u, v


def get_cartesian_coordinates_from_spherical(u, v, r=1) -> np.ndarray:
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    return np.dstack((x, y, z))


def get_cartesian_coordinates(n: int) -> np.ndarray:
    return get_cartesian_coordinates_from_spherical(*get_spherical_coordinates(n))


def plot_sampling_scheme(n: int) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Make points data
    u, v = get_spherical_coordinates(n)
    x, y, z = get_cartesian_coordinates_from_spherical(u, v, 1)

    ax.plot_surface(x, y, z, color='gainsboro', alpha=1.0)
    ax.scatter(x, y, z, marker='o')

    # Set an equal aspect ratio
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Directional visibility map of a point as a sphere')
    plt.savefig('sampling_scheme.png', bbox_inches='tight')


def plot_sampling_grid(n: int) -> None:
    plt.figure()
    u, v = get_spherical_coordinates(n)
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
        plt.plot([0, 2*np.pi], [vv+v[0], vv+v[0]], color='gainsboro')
    plt.savefig('sampling_grid.png', bbox_inches='tight')


if __name__ == '__main__':
    N = 8
    # plot_sampling_scheme(N)
    # plot_sampling_grid(N)
    visibility_map_vectors = get_cartesian_coordinates(N)
    print(visibility_map_vectors.shape)
