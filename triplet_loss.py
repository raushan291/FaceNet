# # # # Will use triplet loss function during training dlib FR model. # # # # 

import dlib
import cv2
import numpy as np


face_detector = dlib.get_frontal_face_detector()
landmarks_predictor = dlib.shape_predictor('/home/rakumar/FaceNet/shape_predictor_68_face_landmarks.dat')
face_encodings_model = dlib.face_recognition_model_v1('/home/rakumar/FaceNet/dlib_face_recognition_resnet_model_v1.dat')


# Triplet loss
def triplet_loss(y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # pos_dist = tf.reduce_sum(tf.square(anchor-positive), axis=-1)
    pos_dist = np.linalg.norm(anchor-positive)

    # neg_dist = tf.reduce_sum(tf.square(anchor-negative), axis=-1)
    neg_dist = np.linalg.norm(anchor-negative)

    basic_loss = np.add(np.subtract(pos_dist, neg_dist), alpha)
    print(basic_loss)

    loss = np.max((0.0, basic_loss))
    
    return loss


anchor_path = '/home/rakumar/FaceNet/train_dir/adpoun/adpoun.1.jpg'
positive_path = '/home/rakumar/FaceNet/train_dir/adpoun/adpoun.2.jpg'
negative_path = '/home/rakumar/FaceNet/train_dir/alebes/alebes.1.jpg'

anchor = cv2.imread(anchor_path)
anchor_faces = face_detector(anchor, 1)
for anchor_face in anchor_faces:
    anchor_landmarks = landmarks_predictor(anchor, anchor_face)
    anchor_face_encodings = np.array(face_encodings_model.compute_face_descriptor(anchor, anchor_landmarks, num_jitters=1))

positive = cv2.imread(positive_path)
positive_faces = face_detector(positive, 1)
for positive_face in positive_faces:
    positive_landmarks = landmarks_predictor(positive, positive_face)
    positive_face_encodings = np.array(face_encodings_model.compute_face_descriptor(positive, positive_landmarks, num_jitters=1))

negative = cv2.imread(negative_path)
negative_faces = face_detector(negative, 1)
for negative_face in negative_faces:
    negative_landmarks = landmarks_predictor(negative, negative_face)
    negative_face_encodings = np.array(face_encodings_model.compute_face_descriptor(negative, negative_landmarks, num_jitters=1))

loss = triplet_loss(y_pred=[anchor_face_encodings, positive_face_encodings, negative_face_encodings])
print('loss=> ',loss)