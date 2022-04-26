import cv2
import os 
import numpy as np

def load_images_from_folder(folder):
    count = 0
    v = sorted(sorted(os.listdir(folder)), key=len)

    for filename in v:
        count+=1
        img = cv2.imread(os.path.join(folder,filename))

        v = np.zeros(img.shape)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if (img[i][j] == np.array([56, 56, 56])).all():
                    v[i][j] = np.array([0, 0, 0])
                elif (img[i][j] == np.array([0, 0, 0])).all() or (img[i][j] == np.array([1, 1, 1])).all():
                    v[i][j] = np.array([255, 255, 255])
                else:
                    v[i][j] = np.array([0, 255, 0])

        cv2.imwrite("drone_basic/correct_output_images/img{}.png".format(count), v)

load_images_from_folder("drone_basic/input_images_2_copy")