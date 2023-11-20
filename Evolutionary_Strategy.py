import numpy as np
from Generation import The_Generation

def Evolutionary_Strategy(input_data,
                          output_data,
                          size_of_population: int = 500,
                          number_of_variables: int = 3,
                          strategy: str = 'include parents' or 'replace parents'):
    # strategy should be defined as follows:
    # strategy = 'include parents'
    # or
    # strategy = 'replace parents'

    generation = The_Generation(input_data, output_data, size_of_population, number_of_variables)

    parents = np.zeros((size_of_population, 4))
    parents[:, :3] = generation.Initial_Population([-10, 10])
    std = np.zeros((size_of_population,4))
    std[:, :3] = generation.Initial_Population([0, 10])

    # calculating MSE for each parent so best_parent can be obtained
    for i in range(size_of_population):
        parents[i, 3] = generation.MSE(generation.The_Function(parents[i,:3]))

    best_parent = parents[np.where(parents == parents[:,3].min())[0],:]
    best_offspring = np.copy(best_parent)*10 # just for first iteration

    t = 0

    while abs(best_parent[0, 3]-best_offspring[0, 3]) > 10**(-5):
        t += 1
        best_parent = parents[0,:].reshape(1,4)
        
        # mutating the parents 5 times - to get 5*size_of_parents chromosomes
        offsprings = np.empty((0, 4))

        for j in range(5):
            mutated_abc = np.zeros((size_of_population, 4))
            mutated_abc[:, :3] = generation.Mutation_of_Chromosomes(parents[:, :3], std[:, :3])
        
            for k in range(len(mutated_abc[:,0])):
                mutated_abc[k, 3] = generation.MSE(generation.The_Function(mutated_abc[k, :3]))

            offsprings = np.concatenate((offsprings, mutated_abc), axis = 0)
        
        best_offspring = offsprings[np.where(offsprings == min(offsprings[:,3]))[0], :]

        if strategy == 'include parents':
            # taking the best u chromosomes from (lambda + u)
            parents_and_offsprings = np.concatenate((parents, offsprings), axis = 0)
            parents = np.copy(parents_and_offsprings[parents_and_offsprings[:,3].argsort()][:size_of_population,:])
            
        elif strategy == 'replace parents':
            # taking the best u chromosomes from lambda
            parents = np.copy(offsprings[offsprings[:,3].argsort()][:size_of_population,:])
            

        std_offsprings = np.empty((0,4))
        
        # mutating the std chromosomes also 5 times
        for l in range(5):
            mutated_std_abc = np.zeros((size_of_population, 4))
            mutated_std_abc[:, :3] = generation.Mutation_of_std(std[:, :3])
            
            std_offsprings = np.concatenate((std_offsprings, mutated_std_abc), axis = 0)

        if strategy == 'include parents':
            the_whole_std = np.concatenate((std,std_offsprings), axis = 0)
            how_influence = generation.Mutation_of_Chromosomes(parents_and_offsprings[:, :3], the_whole_std[:, :3])
        
            for m in range(len(how_influence[:, 0])):
                the_whole_std[m, 3] = generation.MSE(generation.The_Function(how_influence[m]))
        
            # taking the best u chromosemes from (lambda + u)
            std = np.copy(the_whole_std[the_whole_std[:,3].argsort()][:size_of_population, :])
            
        elif strategy == 'replace parents':
            how_influence = generation.Mutation_of_Chromosomes(offsprings[:, :3], std_offsprings[:, :3])
            
            for m in range(len(how_influence[:, 0])):
                std_offsprings[m, 3] = generation.MSE(generation.The_Function(how_influence[m]))
            
            # taking the best u chromosemes from lambda
            std = np.copy(std_offsprings[std_offsprings[:,3].argsort()][:size_of_population, :])
         
            
        print(f'Number of iteration: {t}, MSE: {parents[0,3]}')
        
    return parents[0, :], generation.The_Function(parents[0, :]), t