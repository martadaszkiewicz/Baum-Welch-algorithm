import numpy as np
# import pandas as pd

# # Implementation of the Metropolis-Hastings algorithm for Bayesian inference of the parameters (mean and standard deviation) of a univariate normal distribution

# the_data = np.array(pd.read_csv('Norm_DataSet2.txt', header=None))
# init_mean = 7.0
# init_st_dev = 2.0


def MetropolisHasting_algorithm(the_data, init_mean, init_st_dev):
    
    # initializing the mean and standard deviation:
    old_mean = init_mean
    old_std = init_st_dev
    
    n = len(the_data)           # number of given data points
        
    # containers to store the sampled means and standard deviations
    mean_container = []
    std_container = []
    
    count = 0
    
    # main loop of the Metropolis-Hastings algorithm:
    while True:
        count += 1
        mean_container.append(old_mean)
        std_container.append(old_std)
        
        # computing the log-likelihood of the current parameters:
        sum_for_old_log = 0
        for i,v in enumerate(the_data):
            [v] = v
            sum_for_old_log = sum_for_old_log + (v-old_mean)**2

        old_log = -(n/2)*np.log(2*np.pi) - (n/2)*np.log(old_std**2) - (1/(2*old_std**2))*sum_for_old_log

        # proposing new parameters by sampling from a normal distribution for mean:
        [new_mean] = np.random.normal(old_mean, 2, 1)
        
        # proposing new standard deviation: 
        new_std = old_std+np.random.uniform(-0.5,0.5)       # (simply adding a noise to the current value)
        
        # computing the log-likelihood of the proposed parameters:
        sum_for_new_log = 0
        for i,v in enumerate(the_data):
            [v] = v
            sum_for_new_log = sum_for_new_log + (v-new_mean)**2

        new_log = -(n/2)*np.log(2*np.pi) - (n/2)*np.log(new_std**2) - (1/(2*new_std**2))*sum_for_new_log

        # Metropolis acceptance criterion
        
        # generating a random number 'u' from a uniform distribution between 0 and 1 (to introduce stochasticity to the decision-making process):
        u = np.random.uniform(0,1)
        
        # probability computed as the exponential of the difference between the posterior probabilities (new_log) of the proposed and current parameters (old_log):
        alpha = np.exp(new_log-old_log)
        
        # accepting or rejecting the proposed parameters:
        # (stochastic decision where the algorithm accepts changes that improve the likelihood of the parameters given the data)
        if u < alpha:
            old_mean = np.copy(new_mean)
            old_std = np.copy(new_std)
        
        
        # termination condition
        if count == 10000-1:
            break
    
    # returning the final mean, standard deviation, and the sampled values
    return old_mean, old_std, mean_container, std_container


# old_mean, old_std, mean_container, std_container = MetropolisHasting_algorithm(the_data, init_mean, init_st_dev)
# print(f'\nFinal mean: {old_mean}\nFinal standard deviation: {old_std}')
