import dlib
from PIL import Image
import cv2
import numpy as np
import os
import pickle

Train_model = False

videoCap = cv2.VideoCapture('/home/rakumar/FaceNet/test.mp4') # /home/rakumar/FaceNet/test_1.mp4

face_detector = dlib.get_frontal_face_detector()
landmarks_predictor = dlib.shape_predictor('/home/rakumar/FaceNet/shape_predictor_68_face_landmarks.dat')
face_encodings_model = dlib.face_recognition_model_v1('/home/rakumar/FaceNet/dlib_face_recognition_resnet_model_v1.dat')

training_faces_encoding = {}

font = cv2.FONT_HERSHEY_DUPLEX

def training():
    train_dir = '/home/rakumar/FaceNet/train_dir/'
    training_img_dir_list = sorted(os.listdir(train_dir))

    for name in training_img_dir_list:
        img_list = sorted(os.listdir(train_dir+name))
        enc = []
        for img in img_list:
            image_path = train_dir+name+'/'+img

            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = face_detector(gray, 1)

            if len(faces) == 1:
                for face in faces:
                    x1 = face.left()
                    y1 = face.top()
                    x2 = face.right()
                    y2 = face.bottom()
                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    landmarks = landmarks_predictor(gray, face)

                    for i in range(68):
                        x = landmarks.part(i).x
                        y = landmarks.part(i).y
                        cv2.circle(image, (x, y), 1, color=(255,0,0), thickness=-1)
                    
                    face_encodings = np.array(face_encodings_model.compute_face_descriptor(image, landmarks, num_jitters=1))

                    enc.append(face_encodings)
                    print('=== ', img ,"  ", name)
            else:
                print('NOT SUITABLE FOR TRAINING..')

            cv2.imshow('frames', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        training_faces_encoding[name] = enc

    with open('encodings.dat', 'wb') as f:
        pickle.dump(training_faces_encoding, f)

if Train_model:
    training()

with open('encodings.dat', 'rb') as f:
    trained_faces_encodings = pickle.load(f)


while True:
    _, frame = videoCap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray, 1)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        landmarks = landmarks_predictor(gray, face)

        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 1, color=(255,0,0), thickness=-1)
        
        face_encodings = np.array(face_encodings_model.compute_face_descriptor(frame, landmarks, num_jitters=1))

        min_dist = 1000
        for i, name in enumerate(trained_faces_encodings):
            for each_img_enc in trained_faces_encodings[name]:
                dist = np.linalg.norm(face_encodings - each_img_enc)
                if dist < min_dist :
                    min_dist = dist
                    identity = name
                    
        if min_dist<0.5:
            print(min_dist, identity)
            cv2.rectangle(frame, (x1, y2 - 10), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, identity, (x1 + 3, y2 - 3), font, 0.5, (255, 0, 255), 1)
        
        else:
            print(min_dist, 'UNKNOWN'+' ('+identity+')')
            cv2.rectangle(frame, (x1, y2 - 10), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, 'UNKNOWN', (x1 + 3, y2 - 3), font, 0.5, (0, 0, 225), 1)

    # frame.resize(200,250)
    cv2.imshow('video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
