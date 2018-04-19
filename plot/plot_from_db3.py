#!/usr/bin/python
#coding: utf-8

import pandas as pd
import psycopg2 as pg
import matplotlib.pyplot as plt

db = pg.connect(database="postgres",user="postgres",password="abcd")

data1 = pd.read_sql("select month as m, profit as p1 from sales", con=db, index_col="m")
data2 = pd.read_sql("select month as m, profit -2000 as p2 from sales", con=db, index_col='m')

df = pd.concat([data1,data2],axis=1)

new_df = df[['p1', 'p2']]

new_df.plot()

plt.xlabel("month")
plt.ylabel("profit")
plt.title("monthly-profit")

plt.show()

db.close()
