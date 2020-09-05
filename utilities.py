from pathlib import Path
from datetime import datetime



def check_if_directory(*args) -> None:
    """Check if any given variables are directories"""
    arg: Path
    for arg in args:
        if not isinstance(arg, Path) or not arg.is_dir():
            raise TypeError("{} is not a directory".format(arg))
def rename_images(source: Path) -> None:
    """Rename images in a directory"""
    # not tested
    check_if_directory(source)
    if not source.is_dir():
        raise TypeError("Given path is not a directory")
    count = 0
    previous_datetime_string = ""
    # picture_dir = BASE_DIR / 'database' / 'rename'
    sorted_list = sorted(
        list(source.glob('*')),
        key=lambda picture: picture.stat().st_ctime
    )
    img_path: Path
    for img_path in sorted_list:
        created_datetime = datetime.fromtimestamp(img_path.stat().st_ctime)
        created_datetime_string = created_datetime.strftime("%y%m%d_%H%M%S")
        if created_datetime_string == previous_datetime_string:
            count += 1
            if count > 999:
                previous_datetime_string = created_datetime_string
                raise Exception("Error: count exceeded 999 for {}".format(img_path))
        else:
            count = 0
        rename_string = "{0}_{1}{2}".format(
            created_datetime_string,
            str(count).zfill(3),
            img_path.suffix,
        )
        previous_datetime_string = created_datetime_string
        rename_path = source / rename_string
        if not rename_path.is_file():
            print("{0} -> {1}".format(
                img_path.name,
                rename_path.name,
            ))
            img_path.rename(rename_path)
def move_json(source_dir: Path, target_dir: Path) -> None:
    """Move all json files in a directory into the target directory"""
    # not tested
    check_if_directory(source_dir, target_dir)

    for image in source_dir.glob('*.json'):
        # print(image)
        # print(image.name)
        # print(target_dir)
        target = target_dir.joinpath(image.name)
        # print(target)
        image.replace(target)

if __name__ == "__main__":
    DRIVE_NAME = "F:\\"
    DATA_DIR = Path(DRIVE_NAME).joinpath('data')
    IMAGE_DIR = DATA_DIR.joinpath('images')
    DUMP_DIR = DATA_DIR.joinpath('dump')
    MODEL_DIR = DATA_DIR.joinpath('models')

    TEST_DIR = IMAGE_DIR.joinpath('2017-12-31')

    # Run this cell to remove .json files from the directories in images.
    # for image_folder in IMAGE_DIR.iterdir():
    #     move_json(image_folder, DUMP_DIR)


    # for image_folder in IMAGE_DIR.iterdir():
    #     for image in image_folder.iterdir():
    #         if image.suffix in ['.jpg', '.gif', '.mp4', '.MOV']:
    #             target = DUMP_DIR.joinpath(image.name)
    #             print(target)
    #     break


    # for image in test_dir.iterdir():
    #     target = None
    #     if image.suffix == '.JPG':
    #         target = image.parent.joinpath(image.stem + '.jpg')
    #     elif image.suffix == '.PNG':
    #         target = image.parent.joinpath(image.stem + '.png')
    #     elif image.suffix == '.jpeg':
    #         target = image.parent.joinpath(image.stem + '.jpg')
        
    #     if image.suffix in ['.gif', '.mp4', '.MOV']:
    #         target = DUMP_DIR.joinpath(image.name)
        
    #     if target is not None:
    #         image.rename(target)


    # count_dict = {}
    # for image_folder in IMAGE_DIR.iterdir():
    #     for image in image_folder.iterdir():
    #         # print(image.suffix)
    #         if image.suffix not in count_dict.keys():
    #             print("Adding {} to the count list".format(image.suffix))
    #             count_dict[image.suffix] = 0
    #         count_dict[image.suffix] += 1
    # count_dict



    # for image_folder in IMAGE_DIR.iterdir():
    #     for image in image_folder.iterdir():
    #         target = None

    #         if image.suffix == '.JPG':
    #             target = image.parent.joinpath(image.stem + '.jpg')
    #         elif image.suffix == '.PNG':
    #             target = image.parent.joinpath(image.stem + '.png')
    #         elif image.suffix == '.jpeg':
    #             target = image.parent.joinpath(image.stem + '.jpg')
            
    #         if image.suffix in ['.gif', '.mp4', '.MOV']:
    #             target = DUMP_DIR.joinpath(image.name)
            
    #         if target is not None:
    #             print("{} -> {}".format(image, target))
    #             image.rename(target)
