# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:27:14 2019

@author: Personal
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab 
mpl.style.use('seaborn')
# To have the same random numbers repeated again and again.
#np.random.seed(2785)

mean = 100
sd = 15
N = 1000
binsize = 50

# Data
IQ = np.random.normal(mean, sd, N)

counts, bins, extras = plt.hist(IQ, binsize, facecolor='chocolate', edgecolor='k', label='IQs', density=True)

# An idealised PDF
pdf = mlab.normpdf(bins, mean, sd) # Creates the pdf of normal distribution
plt.figure()
plt.plot(bins, pdf, label='series', color='xkcd:navy blue') # Use this in case of classic style
plt.xlabel('IQ')
plt.ylabel('Count/Fraction')
plt.xticks(bins[::5]) # Pick every 5th element
plt.title('IQ Distribution Histogram')
plt.grid(True)
plt.legend()
plt.savefig("images/histogram.png", format="png", dpi=400)
plt.show()

