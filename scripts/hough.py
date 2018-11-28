import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
#####################
#Return the theta of best lines in an image
#####################
def hough_theta(th):
    edges = cv2.Canny(th, 50, 100, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    dict = {}
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            # print(rho, theta)
            if theta < np.pi/2:
                if theta in dict:
                    dict[theta] += 1
                else:
                    dict[theta] = 0

    return max(dict, key=lambda k:dict[k])


#####################
#Rotate an image with a given theta
#####################
def Rotate(img, theta):
    # if (theta >= np.pi/2):
    #     theta -= np.pi/2
    RotateMatrix = cv2.getRotationMatrix2D(center=(img.shape[1] / 2, img.shape[0] / 2),
                                            angle=(np.pi / 2 - 180 * theta / np.pi), scale=1)
    RotImg = cv2.warpAffine(img, RotateMatrix, (img.shape[1], img.shape[0]))
    return RotImg

#####################
#Calc the hist of an image
#####################
def Hist(img):
    height, width = img.shape
    HistArray = np.zeros(width,dtype=int)
    for i in range(width):
        for j in range(height):
            if img[j][i]:
                flag = 0
            else:
                flag = 1
            HistArray[i] = HistArray[i] + flag
    return HistArray

#####################
#Return a average and min black threshold
#####################
def calc_black_threshold():
    idx = 0
    total = 0
    min_hist = np.inf
    for fn in os.listdir(r'../data/processed/black/'):
        if fn[-3:] != 'jpg': continue
        img = cv2.imread(r'../data/processed/black/'+fn, 0)      # read in gray scale
        ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        hist = sum(Hist(th)[50:200])
        min_hist = hist if hist<min_hist else min_hist
        total += hist
        idx += 1
    return [total/idx, min_hist]

#####################
#Return whether an image is black
#####################
def isBlack(img):
    ret1, img = cv2.threshold(cv2.GaussianBlur(cv2.equalizeHist(img), (3, 3), 0), 127, 255,
                                       cv2.THRESH_BINARY)
    if(sum(Hist(img)[50:200]) > calc_black_threshold()[0]):
        return 1;
    else:
        return 0;



