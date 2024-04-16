#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA


from copy import deepcopy
from random import randint
from PIL import Image

import cv2


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------
    train_image = cv2.imread("./Images/machu_pichu/1.png")
    query_image = cv2.imread("./Images/machu_pichu/2.png")


    height_t, width_t, nc_t = train_image.shape
    height_q, width_q, nc_q = query_image.shape
    # print(query_image.shape)
    # print(train_image.shape)

    # --------------------------------------
    # Execution
    # --------------------------------------

    # Sift features  -----------------------
    sift_detector = cv2.SIFT_create(nfeatures=500)

    t_key_points, t_descriptors = sift_detector.detectAndCompute(train_image, None)
    q_key_points, q_descriptors = sift_detector.detectAndCompute(query_image, None)

    # Draw the keypoints on the images
    train_image_gui = deepcopy(train_image)
    for key_point in t_key_points: # iterate all keypoints
        x, y = int(key_point.pt[0]), int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255),randint(0, 255))
        cv2.circle(train_image_gui, (x,y), 15, color, 1)

    query_image_gui = deepcopy(query_image)
    for key_point in q_key_points: # iterate all keypoints
        x, y = int(key_point.pt[0]), int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255),randint(0, 255))
        cv2.circle(query_image_gui, (x,y), 15, color, 1)

    # Visualization -----------------------
    train_image_gui = cv2.cvtColor(train_image_gui, cv2.COLOR_BGR2RGB)
    imtrain_pil = Image.fromarray(train_image_gui)
    query_image_gui = cv2.cvtColor(query_image_gui, cv2.COLOR_BGR2RGB)
    imquery_pil = Image.fromarray(query_image_gui)
    # cv2.namedWindow('train image', cv2.WINDOW_NORMAL)
    # cv2.imshow('train image', train_image_gui)

    # cv2.namedWindow('query image', cv2.WINDOW_NORMAL)
    # cv2.imshow('query image', query_image_gui)

    # cv2.waitKey(0)
    # --------------------------------------
    # Termination
    # --------------------------------------


if __name__ == "__main__":
    main()