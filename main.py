"""
This is the main file of the face project.
"""
import math
from pathlib import Path
from dataclasses import dataclass

from skimage import io
import matplotlib
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import cv2
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import numpy as np
import pandas as pd
# Import face_alignment last due to a weird OpenMP error
import face_alignment

matplotlib.rcParams['figure.dpi'] = 300
# plt.rcParams['toolbar'] = 'None' # Remove tool bar (upper)


# BASE_DIR = Path(__file__).parent.parent
# BASE_DIR = Path.cwd().
DRIVE_NAME = "F:\\"
DATA_DIR = Path(DRIVE_NAME).joinpath('data')
IMAGE_DIR = DATA_DIR.joinpath('images')
DUMP_DIR = DATA_DIR.joinpath('dump')
MODEL_DIR = DATA_DIR.joinpath('models')
TEST_DIR = DATA_DIR.joinpath('test')
# for c in DATA_DIR.iterdir(): print(c)
# test_dir = IMAGE_DIR.joinpath('2020-08-17')
# test_dir = IMAGE_DIR.joinpath('2020-06-25')
# test_dir = IMAGE_DIR.joinpath('2012-06-11')

@dataclass
class Predtype:
    """
    Data class to simplify landmark data.
    """
    list_slice: slice
    color: tuple
PRED_TYPES = {
    'face': Predtype(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
    'eyebrow1': Predtype(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
    'eyebrow2': Predtype(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
    'nose': Predtype(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
    'nostril': Predtype(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
    'eye1': Predtype(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
    'eye2': Predtype(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
    'lips': Predtype(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
    'teeth': Predtype(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
}

class FaceDetection:
    """
    Class encapsulating bounding box detection data.
    """
    w_1: int
    h_1: int
    w_2: int
    h_2: int
    side_length: int
    confidence: float
    def __init__(self, face_detection: list, multiplier: float):
        w_1 = face_detection[0]
        h_1 = face_detection[1]
        w_2 = face_detection[2]
        h_2 = face_detection[3]
        height = h_2 - h_1
        width = w_2 - w_1
        height_center = h_2 - (h_2 - h_1)/2
        width_center = w_2 - (w_2 - w_1)/2
        if width > height:
            self.side_length = width * multiplier
        else:
            self.side_length = height * multiplier
        self.confidence = face_detection[4]
        self.h_1 = math.floor(height_center - self.side_length/2)
        self.h_2 = math.ceil(height_center + self.side_length/2)
        self.w_1 = math.floor(width_center - self.side_length/2)
        self.w_2 = math.ceil(width_center + self.side_length/2)

class FaceDetector:
    """
    Class encapsulating functions related to the Face Alignment module
    """
    model: face_alignment.FaceAlignment
    def __init__(self):
        self.model = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._3D,
            # face_alignment.LandmarksType._2D,
            face_detector='sfd',
            # face_detector='blazeface',
            flip_input=False,
            device='cpu',
            verbose=True,
        )
    def get_landmarks_and_detections(self, img, sfd_resize_p=480) -> tuple:
        """
        Method to get bounding box detections and landmarks from an image.
        """
        # standard format is (WIDTH x HEIGHT)
        # but img.shape returns it in opposite order
        height, width, _ = img.shape
        if height > width:
            # Vertical image
            multiply_factor = sfd_resize_p/width
            dim = (sfd_resize_p, int(height*multiply_factor))
        else:
            # Horizontal image
            multiply_factor = sfd_resize_p/height
            dim = (int(width*multiply_factor), sfd_resize_p)
        resized_img = cv2.resize(
            img,
            dim,
            interpolation=cv2.INTER_AREA
        )
        # Get lists of landmarks and detections, which are both lists of coordinates
        landmarks, detections = self.model.get_landmarks_from_image(resized_img)
        if landmarks is None and detections is None:
            return None, None
        return (
            [self.rescale_landmark(landmark, multiply_factor) for landmark in landmarks],
            [self.rescale_detection(detection, multiply_factor) for detection in detections],
        )
    @staticmethod
    def rescale_detection(detection: list, multiply_factor: float) -> list:
        """
        Rescales the coordinates in the bounding box detection to the original image dimensions.
        """
        if len(detection) != 5:
            print("Can't rescale a detection that has more than 4+1 items!")
            return detection
        detection[0] = detection[0]/multiply_factor
        detection[1] = detection[1]/multiply_factor
        detection[2] = detection[2]/multiply_factor
        detection[3] = detection[3]/multiply_factor
        return detection
    @staticmethod
    def rescale_landmark(landmark: list, multiply_factor: float) -> list:
        """
        Rescales the coordinates in the landmarks to the original image dimensions.
        """
        return np.array(landmark)/multiply_factor

class EmbeddingExtractor:
    """
    Class encapsulating functions related to the VGGFace module
    """
    model: VGGFace
    def __init__(self):
        self.model = VGGFace(
            # model='resnet50',
            model='senet50',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg',
        )
        print('Inputs: %s' % self.model.inputs)
        print('Outputs: %s' % self.model.outputs)
    def get_embeddings(self, img, detections: list, vgg_resize_p=224, show_resize=False):
        """
        Method to get embeddings from an image using the bounding box detection.
        """
        embeddings = []
        # print(detections)
        for detection in detections:
            # print(detection)
            detection_obj = FaceDetection(detection, multiplier=1.2)
            print("- Confidence: {}".format(round(detection_obj.confidence, 3)))
            face_img = img[
                detection_obj.h_1:detection_obj.h_2,
                detection_obj.w_1:detection_obj.w_2,
            ]

            if detection_obj.side_length > vgg_resize_p:
                interp_method = cv2.INTER_AREA
            else:
                interp_method = cv2.INTER_CUBIC
            face_img = cv2.resize(
                face_img,
                (vgg_resize_p, vgg_resize_p),
                interpolation=interp_method
            )

            if show_resize:
                fig = plt.figure()
                axes = fig.add_axes([0, 0, 1, 1])
                axes.imshow(face_img)
                axes.axis('off')

            samples = preprocess_input(
                np.expand_dims(
                    face_img.astype('float32'),
                    axis=0
                ),
                version=2
            )
            embeddings.append(self.model.predict(samples))
        return embeddings

def get_det_lan_emb_as_df(directory: Path, dir_n: int) -> pd.DataFrame:
    """
    Function returns a dataframe of rows of faces with image paths,
    bounding box detections, landmarks, and embeddings.
    """
    dir_i = 0
    dict_list = []
    face_detector = FaceDetector()
    embedding_extractor = EmbeddingExtractor()

    for image_dir in directory.iterdir():
        if len(list(image_dir.glob("already_processed.txt"))) == 1:
            print("--- Directory already processed ")
            continue
        dir_i += 1
        print("----- Processing images in {}".format(image_dir))
        files = list(image_dir.iterdir())
        f_i = 0
        f_n = len(files)
        for image_file in files:
            if image_file.suffix not in ['.jpg', '.png', '.jfif']:
                print("!!!!! Not an image: {}".format(image_file))
            else:
                f_i += 1
                print("--- Processing {}/{}, {}".format(
                    f_i,
                    f_n,
                    image_file.name
                ))
                img = io.imread(image_file)
                landmarks, detections = face_detector.get_landmarks_and_detections(img)
                if landmarks is None and detections is None:
                    print("--- 0 faces detected, continuing to next image")
                    continue
                print("--- {} faces detected".format(len(detections)))
                embeddings = embedding_extractor.get_embeddings(img, detections)
                for index, embedding in enumerate(embeddings):
                    dict_list.append({
                        'img_path': str(image_file),
                        'detection': detections[index],
                        'landmarks': landmarks[index],
                        'embedding': embedding
                    })
        if dir_i >= dir_n:
            print("----- Finished processing {} directories".format(dir_i))
            break
    if dir_i < dir_n:
        print("!!!!! Finished processing all directories in main directory !!!!!")
    return pd.DataFrame.from_dict(dict_list)

def view_face_data(face_df: pd.DataFrame) -> None:
    """
    Function displays images with extracted face alignment data plotted.
    """
    img_path = None
    plot_style = dict(
        marker='.',
        markersize=1,
        linestyle='none',
        # linewidth=2,
    )

    user_input = input(
        "--- Start index: "
    )

    try:
        index = int(user_input)
    except ValueError:
        print("!!!!! Not an integer !!!!!")
        return

    print("Press any key to continue to the next image. Enter exit to quit.")
    continue_viewing = True
    max_row = len(face_df.index)
    while continue_viewing and index < max_row:
        # Analyze images in dataframe
        row = face_df.iloc[index]

        # Draw image if looking at a different image (or first image) than the last image
        if img_path is None or img_path != row['img_path']:
            if img_path is not None:
                user_input = input()
                plt.close()
                if user_input == "exit":
                    return
            img_path = row['img_path']
            print("Viewing {}".format(Path(img_path).name))
            fig = plt.figure()
            axes = fig.add_axes([0, 0, 1, 1])
            axes.imshow(io.imread(img_path))
            axes.axis('off')

            # fig.canvas.window().statusBar().setVisible(False) # Remove status bar (bottom)
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()

            fig.show()

        detection = FaceDetection(row['detection'], multiplier=1)
        print("Confidence: {}".format(round(detection.confidence, 3)))

        # Draw bounding box
        rect = patches.Rectangle(
            (detection.w_1, detection.h_1),
            detection.side_length,
            detection.side_length,
            linewidth=1,
            edgecolor='g',
            facecolor='none',
        )
        axes.add_patch(rect)
        plt.plot(detection.w_1, detection.h_1, 'ro')
        plt.plot(detection.w_1, detection.h_2, 'go')
        plt.plot(detection.w_2, detection.h_1, 'bo')
        plt.plot(detection.w_2, detection.h_2, 'yo')

        # Draw 2D landmarks
        for pred_type in PRED_TYPES.values():
            axes.plot(
                row['landmarks'][pred_type.list_slice, 0],
                row['landmarks'][pred_type.list_slice, 1],
                color=pred_type.color,
                **plot_style,
            )
        index += 1
    plt.close()

def save_face_df(face_df: pd.DataFrame) -> None:
    """
    Function to save the given dataframe to a pickle file.
    """
    face_df.to_pickle(MODEL_DIR / "dataframe.pickle")
    # add stuff to combine df's and drop in already_processed.txt files into directories

def load_face_df() -> pd.DataFrame:
    """
    Function to load the dataframe from the pickle file.
    """
    return pd.read_pickle(MODEL_DIR / "dataframe.pickle")

def main_loop():
    """
    This is the main loop of the program for the operator to use.
    """
    home_guide = """
    ------- Home -------
    - exit = Quit program
    - load = Load the saved dataframe and do something with it
    - prog1 = Run face alignment and vggface embedding extractor
    """

    continue_program = True
    while continue_program:
        user_input = input(home_guide)
        if user_input == "exit":
            continue_program = False
        elif user_input == "load":
            load()
        elif user_input == "prog1":
            prog1()
        else:
            print("!!!!! Input not recognized, please try again. !!!!!")

def load() -> None:
    """
    Function for loading and viewing options.
    """
    face_df = load_face_df()
    user_input = input(
        "--- Do you want to view?\n"
    )
    if user_input == "view":
        view_face_data(face_df)
    else:
        print("- Returning to home")
def prog1() -> None:
    """
    Function for face data extraction options.
    """
    user_input = input(
        "--- How many folders do you want to process?\n"
    )
    try:
        folder_count = int(user_input)
    except ValueError:
        print("!!!!! Please choose an integer between 1 and 10 !!!!!")
        folder_count = None
    if folder_count is None:
        print("!!!!! Please choose between 1 and 10 !!!!!")
    elif 1 <= folder_count <= 10:
        face_df = get_det_lan_emb_as_df(TEST_DIR, folder_count)
        print("- Processing complete")
        prog1_end(face_df)
    print("- Returning to home")

def prog1_end(face_df: pd.DataFrame) -> None:
    """
    Function for options after face data extraction.
    """
    finalize_prog1 = True
    while finalize_prog1:
        user_input = input(
            "--- Do you want to view, save, or exit (without saving)?\n"
        )
        if user_input == "view":
            view_face_data(face_df)
        elif user_input == "save":
            save_face_df(face_df)
            finalize_prog1 = False
        elif user_input == "exit":
            finalize_prog1 = False
        else:
            print("!!!!! Please try again, or exit !!!!!")










if __name__ == "__main__":
    main_loop()
