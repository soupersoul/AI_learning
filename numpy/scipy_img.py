#!/usr/bin/python
#coding: utf-8

#test1
from scipy.misc import imread, imsave, imresize

from scipy.spatial.distance import pdist, squareform
import numpy as np

def test1():
    img = imread("./pic.jpg")
    print (img.dtype, img.shape)

    img_tinted = img * [0.5, 1, 1]
    imsave('changed_color_pic.jpg', img_tinted)

    img2 = imresize(img_tinted, (500, 500))
    imsave("resized_pic.jpg", img2)

def test2():
    x = np.array([[0,0], [3,0], [3,4]])
    d = squareform(pdist(x, 'euclidean'))
    print d

if __name__ == '__main__':
    #test1()
    test2()
