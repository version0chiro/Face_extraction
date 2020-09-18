import sys
import os
import dlib
import glob
from skimage import io
import numpy as np
import cv2
from imutils import face_utils

def rgbSplitter(frame):
    image = frame
    b = image.copy()
    # set green and red channels to 0
    b[:, :, 1] = 0
    b[:, :, 2] = 0


    g = image.copy()
    # set blue and red channels to 0
    g[:, :, 0] = 0
    g[:, :, 2] = 0

    r = image.copy()
    # set blue and green channels to 0
    r[:, :, 0] = 0
    r[:, :, 1] = 0

    return (b,g,r)


cap = cv2.VideoCapture(0)

# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280, 720))

predictor_path = 'shape_predictor_81_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    (b,g,r)=rgbSplitter(frame)
    dets = detector(frame, 0)
    for k, d in enumerate(dets):
        shape = predictor(frame, d)
        shape = face_utils.shape_to_np(shape)
        # landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
        # for num in range(shape.num_parts):
            # cv2.circle(frame, (shape.parts()[num].x, shape.parts()[num].y), 3, (0,255,0), -1)
        cv2.imshow('test',frame[shape[29][1]:shape[33][1], shape[54][0]:shape[12][0]])
        cv2.rectangle(frame,(shape[69][0],shape[76][1]),(shape[72][0],shape[26][1]),(255,0,0),2)
        cv2.rectangle(frame,(shape[54][0],shape[29][1]), (shape[12][0],shape[33][1]),(255,0,0),2)
        cv2.rectangle(frame,(shape[4][0],shape[29][1]), (shape[48][0],shape[33][1]),(255,0,0),2)
            # cv2.rectangle(frame,shape[29][1]:shape[33][1], shape[4][0]:shape[48][0],(255,0,0),2)

    cv2.imshow('frame', frame)
    cv2.imshow('0', b)
    cv2.imshow('1', g)
    cv2.imshow('2', r)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("q pressed")
        break


cap.release()
out.release()

cv2.destroyAllWindows()
