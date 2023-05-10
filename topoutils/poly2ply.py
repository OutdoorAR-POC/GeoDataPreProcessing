from pathlib import Path
import glob

Color = tuple[int, int, int]
Node = str
color_map = {"Blue": (0, 170, 255), "Yellow": (255, 255, 0), "Red": (255, 0, 0), "Green": (0, 255, 127)}


def create_ply_file_contents(vertices: list[Node], line_color: Color, node_color: Color = (255, 255, 255)) -> list[str]:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element edge {len(vertices) - 1}",
        "property int vertex1",
        "property int vertex2",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    # vertices
    node_color_str = " ".join(str(c) for c in node_color)
    for vertex in vertices:
        lines.append(f"{vertex} {node_color_str}")

    # edges
    line_color_str = " ".join(str(c) for c in line_color)
    for i in range(len(vertices) - 1):
        lines.append(f'{i} {i + 1} {line_color_str}')
    return lines


def save_to_file(contents: list[str], path_to_file: Path) -> None:
    with path_to_file.open('tw') as output_file:
        output_file.write('\n'.join(contents) + '\n')


def read_poly_file_contents(input_path: Path) -> list[Node]:
    with input_path.open('rt') as input_file:
        contents = input_file.readlines()
    return contents


def get_line_color(stem: str) -> Color:
    for color_name, rgb_values in color_map.items():
        if stem.startswith(color_name):
            return rgb_values
    raise Exception("Unknown color")


if __name__ == '__main__':
    folder = Path(r"e:\topo3d\20230509_sobotka_20230320")
    for file in glob.glob('*.poly', root_dir=folder):
        file_stem = file[:-5]
        output_file_path = folder.joinpath(f"{file_stem}.ply")
        points = read_poly_file_contents(folder.joinpath(file))
        line_color = get_line_color(file_stem)
        ply_file_contents = create_ply_file_contents(points, line_color)
        save_to_file(ply_file_contents, output_file_path)
