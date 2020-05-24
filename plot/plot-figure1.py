#!/usr/bin/python
#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,50)
y = x*x
plt.figure(num=6,figsize=(5,10))
plt.plot(x,y)
#plt.show()

y2=2*x+1
plt.figure()
plt.plot(x,y2)
plt.show()

