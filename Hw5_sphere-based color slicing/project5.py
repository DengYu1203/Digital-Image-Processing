import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math

def read_file():
    img = cv2.imread('violet (color).tif',cv2.IMREAD_COLOR) # BGR
    rgb_img = img[:,:,::-1]
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return rgb_img, hsv_img

output_dir = os.path.join('output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def show_img(img,figname):
    plt.figure(figname)
    # plt.imshow(img, cmap ='gray')
    plt.imshow(img)
    path = os.path.join(output_dir,figname+'.png')
    plt.show()
    cv2.imwrite(path,img)
    return

def HSI_split(hsv_img):
    H, S, I = cv2.split(hsv_img)
    show_img(H,'Hue')
    show_img(S,'Saturation')
    show_img(I,'Intensity')
    return

def color_slicing(rgb_img):
    R0 = 30
    a1 = [134, 51, 143]
    a2 = [131, 132, 4]
    # print(rgb_img.shape) # 1024x1024x3
    a1_img = rgb_img.copy()
    a2_img = rgb_img.copy()
    for i in range(rgb_img.shape[0]):
        for j in range(rgb_img.shape[1]):
            r1 = 0
            r2 = 0
            for k in range(3):
                r1 += pow(rgb_img[i,j,k]-a1[k], 2)
                r2 += pow(rgb_img[i,j,k]-a2[k], 2)
            if r1 > R0*R0:
                a1_img[i,j,0] = a1_img[i,j,1] = a1_img[i,j,2] = 0.5
            if r2 > R0*R0:
                a2_img[i,j,0] = a2_img[i,j,1] = a2_img[i,j,2] = 0.5
    a1_bgr = cv2.cvtColor(a1_img, cv2.COLOR_RGB2BGR)
    a2_bgr = cv2.cvtColor(a2_img, cv2.COLOR_RGB2BGR)
    show_img(a1_bgr,'a1 image')
    show_img(a2_bgr,'a2 image')
    show_img(a1_bgr+a2_bgr,'combine')
    return

if __name__ == "__main__":
    rgb_img, hsv_img = read_file()
    HSI_split(hsv_img)
    color_slicing(rgb_img)
    