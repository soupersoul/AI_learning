#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

x_values = np.arange(10)
#x_pos = ["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10"]
x_pos = np.arange(10)
y_values = [1, 2,5,7,3,4,5,6,7,8]

plt.bar(x_values, y_values, align="center", alpha=1)
#plt.xticks(x_values, x_pos)
plt.xlabel("x pixel")
plt.ylabel("y pixel")
plt.title("my bar plot")
plt.show()

