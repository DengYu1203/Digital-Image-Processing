import numpy as np
import cv2
import matplotlib.pyplot as plt
import os.path
import csv

def read_file():
    img = cv2.imread('Bird 1.tif',cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('3xzkY.jpg',cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img,(512,512))
    return img

def show_img(img,figname):
    plt.figure(figname)
    plt.imshow(img, cmap ='gray')
    path = os.path.join('output',figname+'.png')
    plt.show()
    cv2.imwrite(path,img)
    return

def DFT(img,filename):  
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    f_bounded = 20*np.log(np.abs(fshift+1))
    f_img = 255 * f_bounded / np.max(f_bounded)
    f_img = f_img.astype(np.uint8)
    show_img(f_img,filename+' DFT magnitude')

    phase = np.angle(fshift)
    phase += -np.min(phase)
    phase = 255 * phase / np.max(phase)
    phase = phase.astype(np.uint8)
    # print(phase)
    show_img(phase,filename+' DFT phase')
    return f

def iDFT(img):
    f = np.fft.ifft2(img)
    img_back = (np.abs(f))
    # img_back = np.clip(img_back,0,255)
    f_img = img_back - np.min(img_back)
    f_img = 255 * f_img / np.max(f_img)
    f_img = f_img.astype(np.uint8)
    show_img(f_img,'Output with Laplaciain filter')
    return img_back

# def laplacian(img,F):
#     laplacian_img = cv2.Laplacian(img,ddepth=-1,ksize=1)
#     show_img(laplacian_img,'Laplacian image')
#     laplacian_fft = DFT(laplacian_img,'Laplacian image')
    
#     H = -laplacian_fft/F
#     # print(H)
#     # H = H - np.min(H)
#     # H /= np.max(H)
#     # H = np.abs(H)
#     print(H)
#     return 

def fourier_multiply(img1,img2):
    img = img1 * img2
    f_bounded = 20*np.log(np.abs(img+1))
    f_img = 255 * f_bounded / np.max(f_bounded)
    f_img = f_img.astype(np.uint8)
    show_img(f_img,'Using laplacian filter DFT magnitude')
    idft_img = iDFT(img)
    table_output(img1,img)
    return idft_img

def laplacian_H(img):
    row, col = img.shape
    H = np.zeros((img.shape))
    for i in range(int(row/2)):
        for j in range(int(col/2)):
            u = i - int(row/2)
            v = j - int(col/2)
            H[i][j] = -1*(u*u + v*v)
            H[511-i][511-j] = H[i][j]
            H[i][511-j] = H[i][j]
            H[511-i][j] = H[i][j]

    # for i in range(row):
    #     for j in range(col):
    #         H[i][j] = -1*(i*i + j*j)
            
    H += -(np.min(H))
    H = H/np.max(H)
    # print(H)
    show_img(H*255,'H')
    return H

def table_output(F,HF):
    path = os.path.join('output','table.csv')
    with open(path,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Original DFT magnitude','pair(u,v)','After Laplacian DFT magnitude','pair(u,v)'])
        abs_f = np.abs(F)
        abs_hf = np.abs(HF)
        original = np.argsort(abs_f.flatten())
        result = np.argsort(abs_hf.flatten())
        size = len(original)-1
        for i in range(25):
            row = []
            original_pair = [int(original[size-i]/512),int(original[size-i]%512)]
            result_pair = [int(result[size-i]/512),int(result[size-i]%512)]
            row.append((abs_f[int(original_pair[0])][int(original_pair[1])]))
            row.append(original_pair)
            row.append((abs_hf[int(result_pair[0])][int(result_pair[1])]))
            row.append(result_pair)
            writer.writerow(row)
    return

if __name__ == "__main__":
    img = read_file()
    show_img(img,'Original image')
    F = DFT(img,'Original image')
    H = laplacian_H(F)
    idft_img = fourier_multiply(F,H)
    enhance_img = img-idft_img
    enhance_img = np.clip(enhance_img,0,255)
    # enhance_img += -np.min(enhance_img)
    # enhance_img = 255 * enhance_img / np.max(enhance_img)
    # enhance_img = enhance_img.astype(np.uint8)
    show_img(enhance_img,'enhance img')
    
    