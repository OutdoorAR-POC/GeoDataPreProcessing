from pathlib import Path

from outdoorar.sphere_sampling import SamplingScheme

PROJECT_DIR = Path(__file__).parents[1]
RESOURCES_DIR = PROJECT_DIR.joinpath('resources')
ANNOTATIONS_DIR = RESOURCES_DIR.joinpath('annotations')
CAMERAS_DIR = RESOURCES_DIR.joinpath('cameras')
IMAGES_DIR = RESOURCES_DIR.joinpath('capturedImages')
FIGURES_DIR = RESOURCES_DIR.joinpath('figures')
MODELS_DIR = RESOURCES_DIR.joinpath('models')
VISIBILITY_DIR = RESOURCES_DIR.joinpath('visibility')
OUTPUT_DIR = PROJECT_DIR.joinpath('output')


def get_visibility_dir(sampling_scheme: SamplingScheme):
    return VISIBILITY_DIR.joinpath(sampling_scheme.name.lower())
