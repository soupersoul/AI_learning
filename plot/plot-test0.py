#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

def test1():
    x = np.arange(0, 2 * np.pi, 0.1)
    y = np.sin(x)
    plt.plot(x,y)
    plt.show()

def test2():
    x = np.arange(0, 3 * np.pi, 0.1)
    y_sin = np.sin(x)
    y_cos = np.cos(x)
    plt.plot(x, y_sin)
    plt.plot(x, y_cos)
    plt.xlabel("x axis")
    plt.ylabel("y_axis")
    plt.title("test")
    plt.legend(['sin', 'cos'])
    plt.show()

def test_sub_plot():
    x = np.arange(0, 3*np.pi, 0.1)
    y_sin = np.sin(x)
    y_cos = np.cos(x)
    
    plt.subplot(2, 1, 1)
    plt.plot(x, y_sin)
    plt.xlabel("x for sin")
    plt.ylabel("sin(x)")
    plt.title("sin plot")

    plt.subplot(2, 1, 2)
    plt.plot(x, y_cos)
    plt.xlabel("x for cos)")
    plt.ylabel("cos(x)")
    plt.title("cos plot")
    
    plt.show()

def test_plot_img():
    img = imread("pic.jpg")
    img1 = imresize(img, (200, 200))
    img2 = img1 * [0.8, 0.7, 0.63]
    plt.subplot(1, 2, 1)
    plt.imshow(img1)

    plt.subplot(1, 2,2)
    plt.imshow(np.uint8(img2))
    plt.show()

if __name__ == "__main__":
    #test1()
    #test2()
    #test_sub_plot()
    test_plot_img()
