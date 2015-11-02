import matplotlib.pyplot as plt
from pylab import *
import numpy as np

x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# subplots_adjust(hspace=0.000)
number_of_subplots=5

for i,v in enumerate(xrange(number_of_subplots)):
    v = v+1
    ax1 = subplot(1, number_of_subplots,v)
    ax1.plot(x,y, '-')
    ax1.plot(y,x, '+')


plt.show()