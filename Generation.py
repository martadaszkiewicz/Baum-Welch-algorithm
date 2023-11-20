import numpy as np


class The_Generation:
    def __init__(self, 
                 input_data, 
                 output_data, 
                 size_of_initial_population: int, 
                 number_of_chromosomes: int):
        self.input_data = input_data
        self.output_data = output_data
        self.size_of_initial_population = size_of_initial_population
        self.number_of_variables = number_of_chromosomes
        self.size_of_data = len(input_data)
        
    def Initial_Population(self, values_range):
        Initial_Population = np.zeros((self.size_of_initial_population, self.number_of_variables))
        for j in range(self.number_of_variables):
            for i in range(self.size_of_initial_population):
                Initial_Population[i,j] = np.random.uniform(values_range[0], values_range[1])
        
        return Initial_Population 
    
    def The_Function(self, single_chromosome):
        function_output = np.zeros(self.size_of_data)

        for i in range(self.size_of_data):
            function_output[i] = single_chromosome[0]*(np.power(self.input_data[i], 2) - single_chromosome[1]*np.cos(single_chromosome[2]*np.pi*self.input_data[i]))

        return function_output
    
    def MSE(self, obtained_data):
        n = self.size_of_data
        the_sum_inside = []
        
        for i in range(n):
            the_sum_inside.append(np.power((self.output_data[i] - obtained_data[i]),2))
        
        mean_square_error = (1/n)*np.sum(the_sum_inside)
        
        return mean_square_error
    
    def Mutation_of_Chromosomes(self, chromosomes, std_chromosomes):
        mutated_chromosomes = np.zeros((np.shape(chromosomes)))
        
        for i in range(len(chromosomes[:,0])):
            a = chromosomes[i,0] + np.random.normal(scale = std_chromosomes[i,0])
            b = chromosomes[i,1] + np.random.normal(scale = std_chromosomes[i,1])
            c = chromosomes[i,2] + np.random.normal(scale = std_chromosomes[i,2])
            mutated_chromosomes[i,:] = np.array([a, b, c])
        
        return mutated_chromosomes
    
    def Mutation_of_std(self, std_chromosomes):
        n = 6

        t_1 = 1/(np.sqrt(2*n))
        t_2 = 1/(np.sqrt(2*np.sqrt(n)))

        r_std_2 = t_2 * np.random.normal(0,1)
        mutated_std = np.zeros(np.shape(std_chromosomes))
        for i in range(len(std_chromosomes[:,0])):
            for j in range(len(std_chromosomes[i,:])):
                r_std_1 = t_1 * np.random.normal(0,1)
                mutated_std[i,j] = std_chromosomes[i,j] * np.exp(r_std_1) * np.exp(r_std_2)
        
        return mutated_std
    