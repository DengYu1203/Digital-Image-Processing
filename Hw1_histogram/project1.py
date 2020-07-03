import numpy as np
from cv2 import cv2 as cv2
import matplotlib.pyplot as plt
import os.path
import csv

def read_file():
    img = cv2.imread('camellia (mono) 512x512.tif',cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('input',img)
    # cv2.waitKey(0)  # 按下任意鍵離開
    # cv2.destroyAllWindows()
    return img

def plt_histogram(img,name):
    # hist = cv2.calcHist([img],[0],None,[256],[0, 256])
    fig = plt.figure(figsize=(20,8),num=name)
    plt.hist(img.flatten(), 256, [0, 256],edgecolor='blue')
    plt.title(name)
    plt.xlim(0,256)
    out_path = os.path.join('output',name)
    plt.savefig(out_path,format='png')
    plt.show()
    
def calculate_pdf(flatten_img):
    pdf = np.zeros((256,1))
    for i in flatten_img:
        pdf[int(i)] += 1
    pdf /= sum(pdf)
    # print(sum(pdf))
    return pdf

def z_pdf():
    pdf = np.full((256,1),float(1248))
    for i in range(64,191):
        pdf[i] = 800
    pdf /= sum(pdf)
    # print(pdf)
    return pdf

def histogram_specification(in_pdf,z_pdf):
    transform = np.zeros((256,1))    
    # pdf_sum = 0
    # z_index = 0
    # z_sum = z_pdf[0]
    # for i in range(256):
    #     pdf_sum += float(in_pdf[i])
    #     while pdf_sum >= z_sum:
    #         # print(pdf_sum,z_sum,z_index)
    #         z_index += 1
    #         if z_index > 255:
    #             z_index = 255
    #             break
    #         else:
    #             z_sum += z_pdf[z_index]
    #     transform[i] = z_index
    # print(transform)
    s = np.zeros((256,1))
    v = np.zeros((256,1))
    for i in range(256):
        if i == 0:
            s[i] = in_pdf[i]
            v[i] = z_pdf[i]
        else:
            s[i] = s[i-1] + in_pdf[i]
            v[i] = v[i-1] + z_pdf[i]
    z_index = 0
    # print(v)
    for j in range(256):
        while s[j] > v[z_index]:
            z_index += 1
            if z_index > 255:
                z_index = 255
                break
        transform[j] = z_index
        
    return transform , s , v

def transform_output(transform,input_pdf,output_pdf,s,v):
    path = os.path.join('output','table.csv')
    with open(path,'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['r','p(r)','s','v','p(z)','z','','r','z=T(r)'])
        for i in range(len(transform)):
            row = []
            row.append(i)
            row.append(float(input_pdf[i]))
            row.append(float(s[i]))
            row.append(float(v[i]))
            row.append(float(output_pdf[i]))
            row.append(i)
            row.append('')
            row.append(i)
            row.append(int(transform[i]))
            writer.writerow(row)
    return

if __name__ == "__main__":
    input_img = read_file()
    # print(input_img.shape)    # 512x512
    plt_histogram(input_img,'input histogram')
    input_pdf = calculate_pdf(input_img.flatten())
    output_pdf = z_pdf()
    transform,s,v = histogram_specification(input_pdf,output_pdf)
    output_img = input_img

    for i in range(512):
        for j in range(512):
            output_img[i][j] = transform[int(input_img[i][j])]

    plt_histogram(output_img,'output histogram')
    output_path = os.path.join('output','output.png')
    cv2.imwrite(output_path,output_img)
    transform_output(transform,input_pdf,output_pdf,s,v)