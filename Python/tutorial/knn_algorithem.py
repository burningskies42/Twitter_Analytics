import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
import pandas as pd
import random

# example with 2 clusters and another obs. for testing
dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

def k_nearest_neighbors(train_data, predict, k=3):
    if len(train_data) >= k:
        warnings.warn('K was set to a value lesser than total voting groups !')

    distances = []
    for group in train_data:
        for features in train_data[group]:
            # np functions are more effective than manuall (from math module)
            # euc_dist = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))

            # also function for euclidean distance:
            euc_dist = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euc_dist, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    print('Condifence:', confidence)
    return vote_result, confidence

result, condfidence = k_nearest_neighbors(dataset,new_features,k=3)

# same loop as above, but in one line
# [[plt.scatter(j[0],j[1],s=100,color=i) for j in dataset[i]] for i in dataset]
# plt.scatter(new_features[0],new_features[1],color='g',s = 100)
# plt.show()
