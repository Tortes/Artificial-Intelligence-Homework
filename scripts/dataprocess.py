import cv2
from glob import glob
from template import imgtemplate
import hough as hh
import os, os.path

def deleteAllFiles(imgpath):
    for the_file in os.listdir(imgpath):
        file_path = os.path.join(imgpath, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


#####################
#Output pass data parts
#####################
def outputpasspart():
    deleteAllFiles('../data/processed/passdata')
    count = 0
    for fn in glob('../data/dataset/pass/*.jpg'):
        img = cv2.imread(fn, 0)

        for i in range(3):
            img_part = img[300:, (i + 1) * 500:(i + 2) * 500]

            ret1, img_part_bin = cv2.threshold(cv2.GaussianBlur(cv2.equalizeHist(img_part),(3,3),0), 127, 255, cv2.THRESH_BINARY)
            theta1 = hh.hough_theta(img_part_bin)
            img_rot = hh.Rotate(img_part, theta1)

            img_rawout = img_rot[140:1380, imgtemplate(img_rot):imgtemplate(img_rot) + 240]
            img_rawout = cv2.pyrDown(cv2.pyrDown(img_rawout))
            img_rawout_reverse1 = img_rawout[::-1]
            filename = "../data/processed/passdata/%d_%d.jpg" % (count, i)
            filename1 = "../data/processed/passdata/%d_%dr1.jpg" % (count, i)
            cv2.imwrite(filename, img_rawout)
            cv2.imwrite(filename1, img_rawout_reverse1)
        count += 1

#####################
#Output fail data parts
#####################
def outputfailpart():
    deleteAllFiles('../data/processed/faildata')
    count = 0
    for fn in glob('../data/dataset/fail/*.jpg'):
        # divide the filename
        filename = [0, 0, 0]
        if "三" in fn:
            filename = [1, 1, 1]
            continue
        elif "左" in fn:
            filename[0] = 1
        elif "中" in fn:
            filename[1] = 1
        elif "右" in fn:
            filename[2] = 1
        else:
            continue
        # read image
        img = cv2.imread(fn, 0)


        for i in range(3):
            if not filename[i]:
                continue
            img_part = img[300:, (i + 1) * 500:(i + 2) * 500]

            ret1, img_part_bin = cv2.threshold(cv2.GaussianBlur(cv2.equalizeHist(img_part),(3,3),0), 127, 255, cv2.THRESH_BINARY)
            theta1 = hh.hough_theta(img_part_bin)
            img_rot = hh.Rotate(img_part, theta1)

            img_rawout = img_rot[300:1540, imgtemplate(img_rot):imgtemplate(img_rot) + 240]
            img_rawout = cv2.pyrDown(cv2.pyrDown(img_rawout))
            img_rawout_reverse1 = img_rawout[::-1]
            img_rawout_reverse2 = cv2.flip(img_rawout, 1)
            img_rawout_reverse3 = cv2.flip(img_rawout[::-1],0)
            filename = "../data/processed/faildata/%d_%d.jpg" % (count, i)
            filename1 = "../data/processed/faildata/%d_%dr1.jpg" % (count, i)
            filename2 = "../data/processed/faildata/%d_%dr2.jpg" % (count, i)
            filename3 = "../data/processed/faildata/%d_%dr3.jpg" % (count, i)


            cv2.imwrite(filename, img_rawout)
            cv2.imwrite(filename1, img_rawout_reverse1)
            cv2.imwrite(filename2, img_rawout_reverse2)
            cv2.imwrite(filename3, img_rawout_reverse3)
        count += 1

#####################
#Output test data parts
#####################
def outputtestpart():
    deleteAllFiles('../data/processed/testdata')
    count = 0
    for fn in glob('../data/dataset/origin/*.jpg'):
        img = cv2.imread(fn, 0)

        for i in range(3):
            img_part = img[300:, (i + 1) * 500:(i + 2) * 500]

            ret1, img_part_bin = cv2.threshold(cv2.GaussianBlur(cv2.equalizeHist(img_part),(3,3),0), 127, 255, cv2.THRESH_BINARY)
            theta1 = hh.hough_theta(img_part_bin)
            img_rot = hh.Rotate(img_part, theta1)

            img_rawout = img_rot[300:1540, imgtemplate(img_rot):imgtemplate(img_rot) + 240]
            img_rawout = cv2.pyrDown(cv2.pyrDown(img_rawout))
            filename = "../data/processed/testdata/%d_%d.jpg" % (count, i)
            cv2.imwrite(filename, img_rawout)
        count += 1
    if os.path.isfile('../output.txt'):
        os.remove('../output.txt')
    with open('../output.txt', 'a+') as f:
        for fn in glob('../data/processed/testdata/*.jpg'):
            f.write(fn + ' ' + '0' + '\n')


#####################
#Output the training data to txt file
#####################
def outputtxt():
    if os.path.isfile('../train.txt'):
        os.remove('../train.txt')
    with open('../train.txt', 'a+') as f:
        for fn in glob('../data/processed/faildata/*.jpg'):
            if fn.find('3') == -1:
                f.write(fn + ' ' + '1' + '\n')
        for fn in glob('../data/processed/passdata/*.jpg'):
            if fn.find('3') == -1:
                f.write(fn + ' ' + '0' + '\n')

    if os.path.isfile('../val.txt'):
        os.remove('../val.txt')
    with open('../val.txt', 'a+') as f:
        for fn in glob('../data/processed/faildata/*.jpg'):
            if fn.find('3') != -1:
                f.write(fn + ' ' + '1' + '\n')
        for fn in glob('../data/processed/passdata/*.jpg'):
            if fn.find('3') != -1:
                f.write(fn + ' ' + '0' + '\n')

    if os.path.isfile('../output.txt'):
        os.remove('../output.txt')
    with open('../output.txt', 'a+') as f:
        for fn in glob('../data/processed/testdata/*.jpg'):
            f.write(fn + ' ' + '0' + '\n')



if __name__ == "__main__":
    outputtestpart()
    # outputfailpart()
    # outputpasspart()
    # outputtxt()