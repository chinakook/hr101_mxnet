# coding: utf-8
from tiny_fd import TinyFacesDetector
import sys
import cv2 as cv

if __name__ == '__main__':
    detector = TinyFacesDetector(model_root='./', prob_thresh=0.5, gpu_idx=1)
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = './selfie.jpg'
    img = cv.imread(path)
    boxes = detector.detect(img)
    print(boxes.shape)

    for r in boxes:
        cv.rectangle(img, (r[0],r[1]), (r[2],r[3]), (255,255,0),3)
    cv.namedWindow('Tiny FD', cv.WINDOW_NORMAL)
    cv.imshow('Tiny FD', img)
    cv.waitKey()
