
"""
UPDATE SOURCE IF FACE ALIGNMENT IS UPDATED
api.py, return statements for get_landmarks_from_image():

Line 128: return None, None
Line 144: return None, None
Line 186: return landmarks, detected_faces

"""



# img_preds, face_detections = fa.get_landmarks_from_directory(
#     path=str(image_dir),
#     extensions=['.jpg', '.png', '.jfif'],
# )
#    

# 3D-Plot
# for img_path, faces in img_preds.items():
#     # print(img_path)
#     # print(len(value))
#     # print("\n")

#     fig = plt.figure()
#     ax = fig.add_axes([0, 0, 1, 1], projection='3d')
#     # ax.imshow(io.imread(img_path))
#     # ax.axis('off')
#     for index, face in enumerate(faces):
#         # ax = fig.add_subplot(1, len(faces), index + 1)
#         surf = ax.scatter(
#             face[:, 0] * 1.2,
#             face[:, 1],
#             face[:, 2],
#             c='cyan',
#             alpha=1.0,
#             edgecolor='b',
#         )
#         for pred_type in pred_types.values():
#             # print(pred)
#             # print(type(pred))
#             ax.plot3D(
#                 face[pred_type.list_slice, 0] * 1.2,
#                 face[pred_type.list_slice, 1],
#                 face[pred_type.list_slice, 2],
#                 color='blue',
#             )
#         ax.view_init(elev=160., azim=90.)
#         ax.set_xlim(ax.get_xlim()[::-1])
#         plt.show()



# def evaluate(self):
#     """
#     Return ?
#     """
#     # cosine(a, b)

#     # if mtcnn_dict['confidence'] < 0.8:
#     #     print("Confidence: {0} -> skipping a face in {1}".format(
#     #         mtcnn_dict['confidence'],
#     #         image_path,
#     #     ))
#     #     continue
#     return

# def _crop_and_resize(self, base_image, bounding_box):
#     """
#     Returns the face image and the (x, y, w, h) bounding box used.
#     """
#     (x_coord, y_coord, x_width, y_height) = bounding_box
#     x_1 = max(x_coord, 0)
#     x_2 = min(x_coord + x_width, base_image.shape[1])
#     y_1 = max(y_coord, 0)
#     y_2 = min(y_coord + y_height, base_image.shape[0])
#     face_image = base_image[y_1:y_2, x_1:x_2]
#     return (
#         cv2.resize(face_image, (224, 224)),
#         x_1,
#         x_2,
#         y_1,
#         y_2,
#     )
