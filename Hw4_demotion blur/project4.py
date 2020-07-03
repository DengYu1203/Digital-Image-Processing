import numpy as np
import cv2
import matplotlib.pyplot as plt
import os.path
import csv
import math
import cmath

def read_file():
    img = cv2.imread('image-pj4 (motion blurring)2.tif',cv2.IMREAD_GRAYSCALE)
    return img

def show_img(img,figname):
    plt.figure(figname)
    plt.imshow(img, cmap ='gray')
    path = os.path.join('output',str(a)+' '+figname+'.png')
    plt.show()
    cv2.imwrite(path,img)
    return

def show_magnitude(img):
    f_bounded = 20*np.log(np.abs(img)+1)
    f_bounded = f_bounded - np.min(f_bounded)
    f_img = 255 * f_bounded / np.max(f_bounded)
    f_img = f_img.astype(np.uint8)
    return f_img

def DFT(img,filename):  
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    f_img = show_magnitude(fshift)
    show_img(f_img,filename+' DFT magnitude')
    return fshift

def iDFT(img):
    f = np.fft.ifftshift(img)
    f = np.fft.ifft2(f)
    img_back = (np.abs(f))
    f_img = img_back - np.min(img_back)
    f_img = 255 * f_img / np.max(f_img)
    f_img = f_img.astype(np.uint8)
    # show_img(f_img,'Degradation Output image')
    show_img(f_img,'Degradation Output image')
    return f_img

# the direction of linear motion and the displacement
a = 0.0188 # 0.018795
b = 0
T = 1

def restore(H,img):
    reconstructed_img = np.divide(img,H)
    plot_curve(reconstructed_img,H,'reconstructed curve')
    f_img = show_magnitude(reconstructed_img)
    show_img(f_img,'reconstructed DFT magnitude')
    img_back = iDFT(reconstructed_img)
    return 

def uniform_linear_motion_blur(img):
    # print(img.shape)    # 512x512
    row, col = img.shape
    H = np.zeros((img.shape),dtype = np.complex)    
    for i in range(int(row)):
        for j in range(int(col)):
            u = i - int(row/2)
            v = j - int(col/2)
            theta = cmath.pi * ( u * a + v * b )
            if u == 0 and v == 0:
                H[i][j] = T
            elif u == 0:
                H[i][j] = T
            else:
                H[i][j] = (T * cmath.sin(theta) * cmath.exp( -1j * theta ) ) / theta
    # print(H)
    show_img(show_magnitude(H),'H magnitude')
    plot_curve(img,H,'curve')
    restore(H,img)
    
    return H

def blur(H, img):
    blur_img = H * DFT(img,"test")
    img_back = iDFT(blur_img)
    dft_test = DFT(img_back,"test back img")
    H_test = uniform_linear_motion_blur(dft_test)
    return

def plot_curve(img,H,filename):
    x = []
    y = []
    h = []
    f_img = show_magnitude(img)
    f_H = show_magnitude(H)
    for i in range(img.shape[0]):
        x.append(i)
        y.append(f_img[i][256])
        h.append(f_H[i][256])
    plt.figure(filename,figsize=(15,8))
    plt.plot(x,y)
    plt.plot(x,h)
    path = os.path.join('output',str(a)+' '+filename+'.png')
    plt.savefig(path)
    plt.show()
    return

if __name__ == "__main__":
    img = read_file()
    show_img(img,'motion blurring img')
    dft_img = DFT(img,"motion blurring img")
    H = uniform_linear_motion_blur(dft_img)