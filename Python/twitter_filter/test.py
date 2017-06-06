from sklearn.tree import export_graphviz
import pickle

with open('lin_clf.pkl', 'rb') as fid:
   lin_clf = pickle.load(fid)
   fid.close()

with open('svm_clf.pkl', 'rb') as fid:
   svm_clf = pickle.load(fid)
   fid.close()

with open('RF_clf.pkl', 'rb') as fid:
   RF_clf = pickle.load(fid)
   fid.close()

for tree in RF_clf:
   print(tree)
   export_graphviz(tree,feature_names=X.columns,filled=True,rounded=True)