import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans,MeanShift
from sklearn.datasets.samples_generator import make_blobs

style.use('ggplot')
colors = 10 * ["g", "r", "c", "b", "k"]

def genenrate_sample(start,end,size):
    arr = []
    cnt = 0
    for i in range(size):
        new_num = int(random.randrange(start,end))
        arr.append(new_num)
        # np.append(arr,new_num)
        cnt+=1

    arr = np.array(arr)
    return arr

# xs = np.append(genenrate_sample(0,500,500),genenrate_sample(501,1000,500))
# ys = np.append(genenrate_sample(0,500,500),genenrate_sample(501,1000,500))

xs,ys = make_blobs(n_samples=100,n_features=2,centers=5,cluster_std=1)
sample = xs
# sample = np.vstack([xs,ys]).T
np.random.shuffle(sample)

# clf = KMeans(n_clusters=3)
clf = MeanShift()
clf.fit(sample)

centroids = clf.cluster_centers_
labels = set(clf.labels_)
distances =  dict.fromkeys(labels,0)
# print(distances)

for i in sample:
    grp = int(clf.predict([i]))
    plt.scatter(i[0],i[1],c= colors[grp],s=10)
    # print(np.linalg.norm(i - centroids[grp]),distances[grp])
    if np.linalg.norm(i - centroids[grp]) > distances[grp]:
        distances[grp] =  np.linalg.norm(i - centroids[grp])


for i in range(len(centroids)):
    plt.scatter(centroids[i][0],centroids[i][1],marker='x',s=50,c='k')
    # print(centroid)
    circle1 = plt.Circle((centroids[i][0], centroids[i][1]), distances[i], color='r', fill=False)
    plt.gcf().gca().add_artist(circle1)

plt.show()

