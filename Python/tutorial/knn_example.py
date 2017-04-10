'''
Algorithem classifies each observation to its k# closest neighbors.
Number of neighbors should be an odd numbers to avoid ties.
Euclidian distances determine association to a class.
Downside is that calculating euclidian distance for a large dataset is very time consuming.
Doesn't scale well.
'''
import sklearn
import numpy as np
from sklearn import preprocessing,model_selection,neighbors
import pandas as pd
import random
import knn_algorithem as knn

# Clean data
df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace=True)
df.drop(['id'],1,inplace=True) # irrelevant data

# Build train/test data
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print('sklearn result:',accuracy)

# unique new data to test the classifier
example_measures = np.array([
    [4,2,1,1,1,2,3,2,1],
    [4,2,4,1,3,2,3,2,1]
])
example_measures.reshape(len(example_measures),-1)

# testing the classifier
predict_for_example = clf.predict(example_measures)

############################################################
# testing the hand-made knn algo.

# convert all data to floats and shuffles it
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[],4:[]}
test_set =  {2:[],4:[]}

# first 80% of data
train_data = full_data[:-int(test_size*len(full_data))]
# last 20% of data
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote,confidence = knn.k_nearest_neighbors(train_set, data, k = 5)
        if group == vote:
            correct +=1
        total +=1

print('Accuracy:',correct/total)
