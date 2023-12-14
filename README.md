# Optimization algorithms Collection

Collection of optimization algorithms for solving diverse problems is presented. Each has been implemented in python programming language. Below you may find the description of the specific operators.

## Algorithms

### Metropolis-Hasting Algorithm
- [MetropolisHasting_algorithm.py](MetropolisHasting_algorithm.py): Implementation of the Metropolis-Hastings algorithm for Bayesian inference of the parameters (mean and standard deviation) of a univariate normal distribution.
- [MetropolisHasting_visualization.py](MetropolisHasting_visualization.py): Visualizations to aid in understanding and analyzing the Metropolis-Hasting algorithm.
- [datasets](./datasets/): Examplary datasets for M-H algorithm.

### Evolutionary Strategy
- [Generation.py](Generation.py): A Python class for implementing the evolutionary strategy (ES) optimization technique.
- [Evolutionary_Strategy.py](Evolutionary_Strategy.py): Integration of the ES algorithm for solving optimization problems.
- [model3.txt](model3.txt): Exemplary dataset.

### Hidden Markov Model
- [forward_algorithm.py](forward_algorithm.py): Implementation of the forward algorithm for Hidden Markov Models.
- [backward_algorithm.py](backward_algorithm.py): Implementation of the backward algorithm for Hidden Markov Models.
- [BaumWelch_algorithm.py](BaumWelch_algorithm.py): Implementation of the Baum-Welch algorithm for training Hidden Markov Models.

## Description
<!---------------------------------------------------------------- Metropolis-Hasting Algorithm ---------------------------------------------------------------->
### **Metropolis-Hasting Algorithm**
The Metropolis-Hastings algorithm is a Markov Chain Monte Carlo (MCMC) method used for Bayesian inference of parameters. In this specific implementation, the algorithm focuses on estimating the parameters (mean and standard deviation) of a univariate normal distribution.
- **Likelihood Computation**

    The log-likelihood of the current parameters (`old_mean` and `old_std`) given the observed data is computed using the univariate normal distribution:

    $$
    \text{log-likelihood}(\text{data} | \text{mean}, \text{std}) = -\frac{n}{2} \log(2\pi) - \frac{n}{2} \log(\text{std}^2) - \frac{1}{2\text{std}^2} \sum_{i=1}^{n} (x_i - \text{mean})^2
    $$


- **Proposal Step**

    New parameters are proposed by sampling from a normal distribution for the mean `new_mean` and adjusting the standard deviation `new_std` by adding a random noise term:

    $$
    \text{{new\_mean}} \sim \mathcal{N}(\text{{old\_mean}}, 2)
    $$

    $$
    \text{{new\_std}} = \text{{old\_std}} + \text{{uniform}}(-0.5, 0.5)
    $$

- **Acceptance Criterion**

    The Metropolis acceptance criterion decides whether to accept or reject the proposed parameters based on a stochastic decision process. The acceptance probability $\alpha$ is calculated as the exponential of the difference between the log-likelihoods of the proposed and current parameters:

    $$
    \alpha = \exp(\text{{new\_log}} - \text{{old\_log}})
    $$

    A random number $u$ from a uniform distribution between 0 and 1 is generated, and if $u < \alpha$, the proposed parameters are accepted.

- **Termination**

    The algorithm iterates through this process for a specified number of iterations, collecting the sampled means and standard deviations.

- **How to Use**

    The provided Python script demonstrates how to use the Metropolis-Hastings algorithm with an example dataset. The commented lines can be used any time.

<!---------------------------------------------------------------- Evolutionary Strategy ---------------------------------------------------------------->
### **Evolutionary Strategy**

The implementation of Evolution Strategy (ES) to optimize the parameters of a function that models a given dataset. The examplary dataset consists of N = 101 measurements of input and output for a system represented by the function $f(·)$.


The function to be modeled using the following mathematical expression:

$$ f(i) = a (i^2 − b\cos(c\pi i))$$

where $a$, $b$ and $c$ are the parameters whose values are to be find.

- **Optimization Approach**

    The goal of the Evolutionary Strategy is to find optimal values for parameters (a, b, c) that minimize the mean square error (MSE) between the actual output and the output predicted by the function. Both ($\mu$, $\lambda$) and ($\mu + \lambda$) approaches have been implemented, utilizing only the mutation operation as the population varying operator.

- **Basic Functions**

    - *Initial Population*:

        Creates a population of chromosomes with randomly generated values within specified constraints.

    - *Function*:
        
        Calculates the output of the function for a given set of parameters.

    - *MSE*:

        Computes the mean square error between the actual and calculated outputs.

    - *Mutation of Chromosomes*:

        Implements mutation in the population of chromosomes with a given standard deviation.

    - *Mutation of Standard Deviation*:

        Implements mutation in the population of standard deviations.

- **Algorithm Evaluation**

    The algorithm is evaluated until the absolute value of MSE between the best parent and the best offspring is smaller than $10^{-5}$. The operation can be performed by either replacing the population of parents with the best offspring($\mu$, $\lambda$) or considering both ($\mu + \lambda$) parents and offspring populations and selecting the best among them.

- **Output**

    The algorithm returns the best values of chromosomes (parameters a, b, and c), the value of MSE for the function with obtained results, and the number of iterations after which the result is obtained.

- **How to Use**

    To use the Evolutionary Strategy algorithm, run the provided script and follow the commented lines for guidance. Experiment with different configurations to optimize the function for your specific problem.


### **Baum-Welch Algorithm for Hidden Markov Models**

The Baum-Welch algorithm is an expectation-maximization (EM) algorithm used for training Hidden Markov Models (HMMs). It involves iteratively updating the model's parameters to maximize the likelihood of the observed data. This implementation focuses on training an HMM with two states using the Baum-Welch algorithm.

- **Algorithm Details**

    The Baum-Welch algorithm consists of two main steps: Expectation (E-step) and Maximization (M-step).

    - **Expectation (E-step)**
    
        **Forward Algorithm:** 
        Calculates the forward probabilities using the given initial probabilities, transition matrix, emission probabilities and the observed sequence.

        $$
        \alpha_{t,i} = \sum_{i=1}^{N}{\alpha_{t-1}(i) a_{ij} b_{i}(o_{t})}
        $$
                
        **Backward Algorithm:**
        Calculates the backward probabilities, representing the probabilities of transitioning from each state to the next states at each position in the sequence.

        - Initialization
        $$
        \beta_{T(i)} = 1, \quad 1 \leq i \leq N
        $$

        - Recursion
        $$
        \beta_{t(i)} = \sum_{j=1}^{N}{a_{ij} b_{j}(o_{t+1}) \beta_{t+1}(j)}
        $$

        - Termination
        $$
        P(o | \lambda) = \sum_{j=1}^{N}{\pi_{j} b_{j}(o_{1}) \beta_{1}(j)}
        $$

    - **Maximization (M-step)**
    
        **Update Transition Matrix ($A$)** 
        Updates the transition matrix based on the forward and backward probabilities.

        **Update Emission Probabilities ($B$)** 
        Updates the emission probabilities based on the forward and backward probabilities.

        **Normalization and Standardization:**
        Normalizes and standardizes the updated matrices.

    - **Convergence and Iteration:**
        The algorithm iterates through the E-step and M-step until the distance between current and previous estimates falls below a specified threshold ($\Delta$).

- **How to Use**

    An exemplary run is provided within the commented lines in the script to demonstrate its usage with sample data.

## Usage
To use any of these algorithms, follow the instructions in the respective algorithm's section or search the commented lines in specific file. Each algorithm comes with example code and usage guidelines to help integrate it into projects.


