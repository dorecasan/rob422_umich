import numpy as np
import matplotlib.pyplot as plt
import time
import random
from sgd import *

maxi = 10000 #this is the number of functions

def fi(x,i):
    coef1 = 0.01 + (0.5-0.01)*i/maxi
    coef2 = 1 + (6-1)*i/maxi
    return (np.exp(coef1*x + 0.1) + np.exp(-coef1*x - 0.5) - coef2*x)/(maxi/100)

def fiprime(x,i):
    coef1 = 0.01 + (0.5-0.01)*i/maxi
    coef2 = 1 + (6-1)*i/maxi
    return (coef1*np.exp(coef1*x + 0.1) -coef1*np.exp(-coef1*x - 0.5) - coef2)/(maxi/100)

def fiprimeprime(x,i):
    coef1 = 0.01 + (0.5-0.01)*i/maxi
    #coef2 = 1 + (6-1)*i/maxi
    return (coef1*coef1*np.exp(coef1*x + 0.1) +coef1*coef1*np.exp(-coef1*x - 0.5))/(maxi/100)


def fsum(x):
    sum = 0
    for i in range(0,maxi):
       sum = sum + fi(x,i)
    return sum

def fsumprime(x):
    sum = 0
    for i in range(0,maxi):
       sum = sum + fiprime(x,i)
    return sum

def fsumprimeprime(x):
    sum = 0
    for i in range(0,maxi):
       sum = sum + fiprimeprime(x,i)
    return sum

#this is just to see the function, you don't have to use this plotting code
xvals = np.arange(-10, 10, 0.01) # Grid of 0.01 spacing from -10 to 10
yvals = fsum(xvals) # Evaluate function on xvals
plt.figure()
plt.plot(xvals, yvals) # Create line plot with yvals against xvals

#this is the timing code you should use
start = time.time()
print("Hello world!")
#YOUR ALGORITHM HERE#
x_values = sgd(fiprime,maxi)
# Plot fsum(x(i)) vs. i
i_values = np.arange(len(x_values))
fsum_values = [fsum(x_i) for x_i in x_values]

plt.figure()
plt.plot(i_values, fsum_values)
plt.xlabel('i')
plt.ylabel('fsum(x(i))')

end = time.time()
print("Time: ", end - start)

mean_var_calculation(fsum,fiprime, maxi, num_runs = 30)

plt.show() #show the plot

