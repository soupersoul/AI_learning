#!/usr/bin/python
#-*- coding: utf-8 -*- 

import psycopg2 as pg
import numpy as np
import matplotlib.pyplot as plt

db=pg.connect(database="postgres", user="postgres", password="abcd", host="127.0.0.1", port="5432")

cur = db.cursor()
cur.execute("select year, month, profit from sales")
rows = cur.fetchall()

dtypes = np.dtype([('year','i2'),('month','i2'),('profit','i4')])
data = np.fromiter(rows, dtype=dtypes, count=-1)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False

plt.bar(data['month'],data['profit'],align="center")
plt.xlabel(u'月份')
plt.ylabel('profit')
plt.title('plot from db')
plt.xticks(data['month'])

plt.show()

db.close()
