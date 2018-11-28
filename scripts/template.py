import cv2

def imgtemplate(img):
    imgtemplate = cv2.imread('/home/tortes/pycharm_program/Image/data/processed/template.jpg',0)
    w, h = imgtemplate.shape[::-1]
    method = eval('cv2.TM_SQDIFF_NORMED')
    res = cv2.matchTemplate(img, imgtemplate, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return min_loc[0]
