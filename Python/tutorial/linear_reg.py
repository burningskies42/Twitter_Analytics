from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

'''
     E(x)*E(y) - E(x*y)
m =  ------------------
      (E(x))^2 - E(x^2)
      
n = E(y)-m*E(x)
'''
'''
Squared Errors for goodness of fit,
coefficient of determination:

              SE(y_hat)     *standart error of y_hat has
  R^2  = 1 - ---------       to be much smaller then E(y)
              SE(E(Y))       to get a high R^2


'''

# Example data:
# xs = np.array([1,2,3,4,5,6], dtype=np.float64)
# ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dateset(n, variance, step=2, corr=False):
    val = 1
    ys = []
    for i in range(n):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if corr and corr == 'pos':
            val += step
        elif corr and corr == 'neg':
            val -= step

    xs=[i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys,dtype=np.float64)

# Gives slope for x,y datasets
def best_lin_fit(xs,ys):
    Ex = mean(xs)
    Ex2 = mean(xs**2)
    E2x = mean(xs)**2
    Ey = mean(ys)
    Ey2 = mean(ys ** 2)
    E2y = mean(ys) ** 2
    Exy = mean(xs * ys)
    m = ( ( ( Ex * Ey ) - Exy )/
          (E2x - Ex2 ))
    n = Ey- (m*Ex)
    return m, n

def sqr_error(ys_org,ys_line):
    return sum((ys_line-ys_org)**2)

def coef_detr(ys_org,ys_line):
    y_mean_line =[mean(ys_org) for y in ys_org]
    squared_error_reg = sqr_error(ys_org,ys_line)
    squared_error_y_mean = sqr_error(ys_org,y_mean_line)
    return 1-(squared_error_reg/squared_error_y_mean)

xs, ys = create_dateset(40,40,2,corr ='neg')

m,n = best_lin_fit(xs,ys)

reg_line = [(m*x)+n for x in xs]
R2 = coef_detr(ys,reg_line)
print(R2)

plt.scatter(xs,ys)
plt.plot(xs,reg_line,color='r')
plt.show()


