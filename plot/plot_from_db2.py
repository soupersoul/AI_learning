#!/usr/bin/python
#coding: utf-8

import psycopg2 as pg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

db=pg.connect(host="localhost",port="5432",user="postgres",password="abcd",database="postgres")

data1=pd.read_sql("select month as m1, profit as p1 from sales",con=db)
data2=pd.read_sql("select month as m, profit + 1050 as p2 from sales", con=db)

dataframe = pd.concat([data1, data2], axis=1)

dataframe_cols = dataframe[['m','p1', 'p2']]

df = dataframe_cols.set_index('m')

#line
#df.plot()
df.plot(kind="bar")
#plt.xticks(dataframe['m'])
plt.xlabel('month')
plt.ylabel('profit')
plt.title('monthly profit')
plt.show()

db.close()
