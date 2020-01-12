from pathlib import Path

# Note: stem + suffix = name
BASE_DIR = Path(__file__).parent.parent
DRIVE_NAME = "F:\\"
DATA_DIR = Path(DRIVE_NAME).joinpath('data')
IMAGE_DIR = DATA_DIR.joinpath('images')
DUMP_DIR = DATA_DIR.joinpath('dump')
MODEL_DIR = DATA_DIR.joinpath('models')
# for c in DATA_DIR.iterdir(): print(c)

IMAGE_FILE_SUFFIX_LIST = [
    '.jpg',
    '.jpeg',
    '.jfif',
    '.png',
]
IMAGE_FILE_SUFFIX_WHITELIST_LIST = [
    '.pkl',
]
