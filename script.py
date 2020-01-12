import pickle
# from datetime import datetime
# from time import time

import cv2
# import torch
import numpy as np
# from scipy.spatial.distance import cosine
from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

import const

### Class
class Extractor(object):
    """
    Returns class extractor object used to extract face data.
    """
    def __init__(self):
        self.mtcnn_face_detector = MTCNN()
        self.vgg_embedding_extractor = VGGFace(
            # model='resnet50',
            model='senet50',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg'
        )
        # print('Inputs: %s' % self.vgg_embedding_extractor.inputs)
        # print('Outputs: %s' % self.vgg_embedding_extractor.outputs)
    def extract(self, image_directory_path):
        """
        Saves face data from images in a directory into a pickle file inside the directory.
        """
        ### Get list of face data
        images_list = self._extract_from_directory(image_directory_path)

        ### Save list of face data as pickle
        self.save(image_directory_path, images_list)

        ### Display extraction process
        self.display_image(images_list)
    def save(self, image_directory_path, images_list):
        """
        Saves face data into a pickle file inside the directory
        """
        pkl_path = self._get_pkl_path(image_directory_path)
        with open(pkl_path, 'wb') as pkl_file:
            pickle.dump(images_list, pkl_file)
    def load(self, image_directory_path):
        """
        Loads face data from a pickle file inside the directory
        """
        pkl_path = self._get_pkl_path(image_directory_path)
        with open(pkl_path, 'rb') as pkl_file:
            return pickle.load(pkl_file)
    def evaluate(self):
        """
        Return ?
        """
        # cosine(a, b)

        # if mtcnn_dict['confidence'] < 0.8:
        #     print("Confidence: {0} -> skipping a face in {1}".format(
        #         mtcnn_dict['confidence'],
        #         image_path,
        #     ))
        #     continue
        return
    def display_image(self, images_list, speed=2):
        """
        Displays base images and face images using opencv.
        Speed is in seconds.
        """
        # print(images_list)
        for image_dict in images_list:
            # print(image_dict)
            base_image = cv2.imread(str(image_dict['path']))
            base_image_for_view = base_image.copy()
            for face_index, face_dict in enumerate(image_dict['face_list']):
                face_image, x_1, x_2, y_1, y_2 = \
                    self._crop_and_resize(base_image, face_dict['mtcnn_extraction']['box'])
                ### Show resized face images along top row of screen
                face_image_window = '{}'.format(face_index)
                cv2.imshow(face_image_window, face_image)
                cv2.moveWindow(face_image_window, 224*face_index, 0)
                ### Draw bounding box on original image
                base_image_for_view = cv2.rectangle(
                    base_image_for_view,
                    (x_1, y_1),
                    (x_2, y_2),
                    (0, 255, 0), # line color
                    2, # line pixel thickness
                )
            ### Show base image underneath the extracted faces
            cv2.imshow('Base image', base_image_for_view)
            cv2.moveWindow('Base image', 0, 224)
            cv2.waitKey(speed*1000)
            cv2.destroyAllWindows()
    def _check_if_image(self, file_path):
        is_image = file_path.suffix in const.IMAGE_FILE_SUFFIX_LIST
        is_white_list = file_path.suffix in const.IMAGE_FILE_SUFFIX_WHITELIST_LIST
        if not is_image and not is_white_list:
            raise TypeError("Unknown file type found: {}".format(
                file_path.suffix,
            ))
        return is_image
    def _get_pkl_path(self, directory_path):
        """
        Returns path of the pickle file containing face data.
        Assumes that the pickle file has the same file name as the directory it resides in.
        """
        return directory_path.joinpath(directory_path.name + '.pkl')
    def _extract_from_directory(self, image_directory_path):
        """
        Returns a list of dicts that contains face data extracted from images in a directory.
        """
        images_list = []
        file_list = list(image_directory_path.iterdir())
        n_file = len(file_list)
        print("Processing images from:\n{}".format(image_directory_path))
        for file_index, file_path in enumerate(file_list):
            print("Processing file {0}/{1}".format(
                file_index,
                n_file,
            ))
            if not self._check_if_image(file_path):
                continue
            face_list = self._extract_from_image(file_path)
            images_list.append({
                'path': file_path,
                'face_list': face_list,
            })
        return images_list
    def _extract_from_image(self, image_path):
        """
        Returns a list of face data extracted from an image (path).
        """
        face_list = []
        ### Load image and use MTCNN to get face bounding box
        base_image = cv2.imread(str(image_path))
        for mtcnn_dict in self.mtcnn_face_detector.detect_faces(base_image):
            ### Get face image from bounding box, preprocess, and extract VGG embeddings
            face_image, _, _, _, _ = self._crop_and_resize(base_image, mtcnn_dict['box'])
            samples = preprocess_input(
                np.expand_dims(
                    face_image.astype('float32'), axis=0
                ),
                version=2,
            )
            embedding = self.vgg_embedding_extractor.predict(samples)
            ### Save data in a dict
            face_list.append({
                'source_image_path': image_path,
                'mtcnn_extraction': mtcnn_dict,
                'vgg_extraction': embedding,
            })
        return face_list
    def _crop_and_resize(self, base_image, bounding_box):
        """
        Returns the face image and the (x, y, w, h) bounding box used.
        """
        (x_coord, y_coord, x_width, y_height) = bounding_box
        x_1 = max(x_coord, 0)
        x_2 = min(x_coord + x_width, base_image.shape[1])
        y_1 = max(y_coord, 0)
        y_2 = min(y_coord + y_height, base_image.shape[0])
        face_image = base_image[y_1:y_2, x_1:x_2]
        return (
            cv2.resize(face_image, (224, 224)),
            x_1,
            x_2,
            y_1,
            y_2,
        )

### Script
if __name__ == "__main__":
    # temp_dir = const.IMAGE_DIR.joinpath('2010-09-27')
    temp_dir = const.IMAGE_DIR.joinpath('2013-12-23')
    extractor = Extractor()

    # extractor.extract(temp_dir)

    image_list = extractor.load(temp_dir)
    extractor.display_image(image_list, speed=1)

    # image_list = extractor.load(temp_dir)
    # a = cv2.imread(str(image_list[0]['path']))
    # b = a
    # a = cv2.rectangle(
    #     a,
    #     (0, 0),
    #     (200, 200),
    #     (0, 255, 0), # line color
    #     2, # line pixel thickness
    # )
    # cv2.imshow('a', a)
    # cv2.imshow('b', b)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()