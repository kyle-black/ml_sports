from scipy.stats import geom
import matplotlib.pyplot as plt
import numpy as np

# Specify the probability of success
p = 0.4865

# Generate a range of numbers for x values
x = np.arange(1, 11, 1)

# Calculate corresponding geometric probabilities
y = geom.pmf(x, p)
z= geom.cdf(x,p)


print(x,y,z)
#plt.bar(x, y)
#plt.title('Geometric Distribution (p=0.2)')
#plt.xlabel('Number of Trials')
#plt.ylabel('Probability')
#plt.show()