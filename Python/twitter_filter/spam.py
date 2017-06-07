import pandas as pd
import pickle

with open('labeled_featureset.pkl','rb') as fid:
   df = pickle.load(fid)
   fid.close()
