#coding: utf-8
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

def show_binary_image(img_arr):
    plt.imshow(img_arr, cmap='binary')
    plt.show()

def show_subplot_images(n_rows, n_cols, imgs, names):
    assert n_rows * n_cols < len(imgs)
    plt.figure(figsize=(n_rows * 2.4, n_cols * 1.2))
    for row in range(n_rows):
        for col in range(n_cols):
            index = row * n_cols + col
            plt.subplot(n_rows, n_cols, index + 1)
            plt.imshow(imgs[index], cmap='binary', interpolation="nearest")
            plt.axis("off")
            plt.title(names[index])
    plt.show()

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=[8, 5])
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
