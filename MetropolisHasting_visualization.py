import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MetropolisHasting_algorithm import MetropolisHasting_algorithm

example_data = np.array(pd.read_csv('./datasets/Norm_DataSet5.txt', header=None))
init_mean = 7.0
init_st_dev = 2.0

final_mean, final_std, mean_container, std_container = MetropolisHasting_algorithm(example_data, init_mean, init_st_dev)

print(f'\nFinal mean: {final_mean}\nFinal standard deviation: {final_std}')

### HISTOGRAM OF ORIGINAL DATA WITH FITTED CURVE OF OBTAINED PARAMETERS ###  

plt.figure(figsize=(8, 6))
plt.hist(example_data, bins=30, density=True, alpha=0.6, color='g',edgecolor='k', label='Histogram of given data')

# fitted curve:
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-0.5 * ((x - final_mean) / final_std) ** 2) / (final_std * np.sqrt(2 * np.pi))
plt.plot(x, p, 'r', linewidth=2, label='Fit: mean=%.2f, std=%.2f' % (final_mean, final_std))

plt.title('Fit results: mean = %.2f,  std = %.2f' % (final_mean, final_std))
plt.legend(loc='upper left')
plt.grid(True, which='major', axis='both')
plt.show()

### OBTAINED PARAMETER VALUE FOR EACH ITERATION + PRESENTATION IN FORM OF HISTOGRAM ###
plt.figure(figsize=(8,4))

plt.subplot(1, 2, 1)
plt.hist(mean_container, bins=30, density=True, alpha=0.6, color='r',edgecolor='k')
plt.ylabel('Count')
plt.xlabel('Iteration')
plt.title('Mean')
plt.grid(True, which='major', axis='both')

plt.subplot(1, 2, 2)
plt.plot(mean_container, color='b')
plt.axhline(y=np.mean(mean_container), color='k', linestyle='--', label='Average = %.2f' % (np.mean(mean_container)))
plt.ylabel('Estimated value')
plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.title('Mean')
plt.grid(True, which='major')

#######################################################################################
plt.figure(figsize=(8,4))

plt.subplot(1, 2, 1)
plt.hist(std_container, bins=30, density=True, alpha=0.6, color='r',edgecolor='k')
plt.ylabel('Count')
plt.xlabel('Iteration')
plt.title('Standard deviation')
plt.grid(True, which='major', axis='both')

plt.subplot(1, 2, 2)
plt.plot(std_container, color='b')
plt.axhline(y=np.mean(std_container), color='k', linestyle='--', label='Average = %.2f' % (np.mean(std_container)))
plt.ylabel('Estimated value')
plt.xlabel('Iteration')
plt.legend(loc='lower right')
plt.title('Standard deviation')
plt.grid(True, which='major')


