from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

'''
     E(x)*E(y) - E(x*y)
m =  ------------------
      (E(x))^2 - E(x^2)
      
n = E(y)-m*E(x)
'''

# Example data:
xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

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

m,n = best_lin_fit(xs,ys)

reg_line = [(m*x)+n for x in xs]

predict_x = 8
predict_y = m*predict_x + n


plt.scatter(xs,ys)
plt.scatter(predict_x,predict_x, color = 'g')
plt.plot(xs, reg_line)
plt.show()
