import pandas as pd
import quandl
import pickle
import tensorflow
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
ax1 = plt.subplot2grid((2,1),(0,0))
ax2 = plt.subplot2grid((2,1),(1,0),sharex = ax1)

bridge_height = {'meters':[10.26, 10.31, 10.27, 10.22, 10.23, 6212.42, 10.28, 10.25, 10.31]}
df = pd.DataFrame(bridge_height)
df['STD'] = df['meters'].rolling(2).std()

df_std = df.describe()['meters']['std']
df = df[(df['STD']<df_std)]
print(df)

print(df_std)
df['meters'].plot(ax = ax1)
ax1.legend(loc = 4)
df['STD'].plot(ax = ax2)
plt.legend(loc = 4)
plt.show()