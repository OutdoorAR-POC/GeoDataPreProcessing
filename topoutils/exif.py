from pathlib import Path
from typing import Any

import PIL.Image
import PIL.ExifTags
import filetype


def read_exif(img_path: Path) -> dict[str, Any]:
    img = PIL.Image.open(img_path)

    exif = {
        PIL.ExifTags.TAGS[k]: v
        for k, v in img._getexif().items()
        if k in PIL.ExifTags.TAGS
    }

    return exif


def get_photo_lat_lon_alt(exif_data: dict[str, Any]):
    gps_info_key = 'GPSInfo'
    exif_data[gps_info_key]


if __name__ == '__main__':
    folder = Path(r'e:\topo3d\zdj\20230320_GEO')
    for file in folder.iterdir():
        if filetype.is_image(file):
            read_exif(file)
