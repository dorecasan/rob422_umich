import random
import matplotlib.pyplot as plt
import numpy as np 

def sgd(fiprime, maxi, initial_x=-5, learning_rate=1.0, epochs=1000):
    
    def gradient(x):
        i = random.randint(0, maxi - 1)
        return fiprime(x, i)

    x_values = [initial_x]
    x = initial_x
    for epoch in range(epochs):
        grad = gradient(x)
        x = x - learning_rate * grad
        x_values.append(x)
    return x_values

def mean_var_calculation(fsum, fiprime, maxi, num_runs = 30):
    fsum_results = []

    for _ in range(num_runs):
        x_values = sgd(fiprime, maxi)
        final_x = x_values[-1]
        fsum_results.append(fsum(final_x))

    # Compute mean and variance
    mean_fsum = np.mean(fsum_results)
    variance_fsum = np.var(fsum_results)

    print(f"Mean of fsum(x*): {mean_fsum}")
    print(f"Variance of fsum(x*): {variance_fsum}")