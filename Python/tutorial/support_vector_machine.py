import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
    def __init__(self, visualisation=True):
        self.visualisation = visualisation
        self.color = {1:'r',2:'b'}
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)

    # train
    def fit(self, data):
        self.data = data
        # {||w|| : [w,b]} - magnitude as key
        opt_dict = {}

        # for each opt. step we must test all transformations
        # for the dot-product
        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        # to get max/min ranges
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001]

        # extremly expensive
        b_range_multiple = 5

        b_multiple = 5
        latest_optimum = self.max_feature_value*10

        # begin stepping:
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])

            # we can do this, because the problem is convex
            # sure to find a global extremum
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        # weakest poin in SVM
                        # SMO imporves this a bit
                        for i in self.data:
                            for x_i in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                    if found_option:
                        opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] <:
                    optimized = True
                    print('optimized a step')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2

    def predict(self, features):
        # sign(x.w+b)
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        return classification

data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8], ]),

             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3], ])}