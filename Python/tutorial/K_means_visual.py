import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import style
from sklearn.cluster import KMeans

style.use('fivethirtyeight')
colors = 10*["g","r","c","b","k"]


# xs = np.random.randint(0,50,50)
# xs = np.append(np.random.randint(0,50,50),np.random.randint(50,100,50))
# ys = np.append(np.random.randint(0,50,50),np.random.randint(50,100,50))
xs = np.random.randint(0,1000,1000)
ys = np.random.randint(0,1000,1000)

obs = np.append(xs,ys).reshape((-1,2))


clf = KMeans(n_clusters=2)
clf.fit(obs)

centroids = clf.cluster_centers_
labels = set(clf.labels_)
# print(labels)

distances = {'1':0,'0':0}

for x,y in obs:
    grp = int(clf.predict([[x,y]]))
    points = plt.scatter(x,y,c=colors[grp],s=10)
    dist = np.linalg.norm([x,y] - centroids[grp])
    if dist > distances[str(grp)]:
        distances[str(grp)] = dist

circles=[]


for i in range(len(centroids)):
    plt.scatter(centroids[i][0],centroids[i][1],s=150,c='k',marker='x')
    circle = plt.Circle((centroids[i][0],centroids[i][1]), radius=distances[str(i)]*1.1, fc='y',fill=False,linewidth=5,color=colors[i])
    plt.gca().add_patch(circle)


# print(distances)
plt.show()