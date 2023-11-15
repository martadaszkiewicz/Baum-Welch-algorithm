import numpy as np
from forward_algorithm import forward_algorithm
from backward_algorithm import backward_algorithm

# import pandas as pd

# # exemplary data: 
# init_prob = np.array([0.53, 0.47])                                      # Initial probabilities for the hidden states
# init_A = np.array(([0.75, 0.25], [0.8, 0.2]),dtype=np.float64)          # Initial transition matrix

# # emit_prob_s = [A T C G]
# # B = [[emit_prob_s1],[emit_prob_s2]]
# B = np.array(([0.2, 0.4, 0.3, 0.1], [0.2, 0.2, 0.5, 0.1]))              # Matrix representing emission probabilities
# sequence = [2, 2, 0, 2, 3, 1, 1, 3]                                     # The observed sequence [C, C, A, C, G, T, T, G]
# delta = 10**(-9)                                                        # Convergence threshold


def BaumWelch_algorithm(init_prob,init_A,B,sequence,delta):
    old_B = np.copy(B)
    old_A = np.copy(init_A)
    
    index_A = [i for i, v in enumerate(sequence) if v == 0]
    index_T = [i for i, v in enumerate(sequence) if v == 1]
    index_C = [i for i, v in enumerate(sequence) if v == 2]
    index_G = [i for i, v in enumerate(sequence) if v == 3]
    indices_cell = (index_A,index_T,index_C,index_G)

    T = len(sequence)-1
    iteration = 0
    while True:
        iteration += 1
        # EXPECTATION:
        alpha_prob = forward_algorithm(init_prob, old_A, sequence, old_B)
        beta_prob = backward_algorithm(old_A, sequence, old_B)

        # MAXIMIZATION:
        # new transition matrix A:
        new_A = np.zeros_like(old_A)
        for t in range(0,T):
            new_A[0,0] = new_A[0,0] + alpha_prob[0,t]*old_A[0,0]*old_B[0,sequence[t+1]]*beta_prob[0,t+1]
            new_A[0,1] = new_A[0,1] + alpha_prob[0,t]*old_A[0,1]*old_B[1,sequence[t+1]]*beta_prob[1,t+1]
            new_A[1,0] = new_A[1,0] + alpha_prob[1,t]*old_A[1,0]*old_B[0,sequence[t+1]]*beta_prob[0,t+1]
            new_A[1,1] = new_A[1,1] + alpha_prob[1,t]*old_A[1,1]*old_B[1,sequence[t+1]]*beta_prob[1,t+1]

        # new current emission probabilities B:
        new_B = np.zeros_like(old_B)
        
        alpha_beta = np.zeros_like(alpha_prob)
        for ab_col in range(0,len(alpha_prob[0,:])):
            for ab_row in range(0,len(alpha_beta[:,0])):
                alpha_beta[ab_row,ab_col] = alpha_prob[ab_row,ab_col]*beta_prob[ab_row,ab_col]
        
        for i, indices in enumerate(indices_cell):
            for j in range(len(indices)):
                new_B[0,i] = new_B[0,i] + alpha_beta[0,indices[j]]
                new_B[1,i] = new_B[1,i] + alpha_beta[1,indices[j]]
        
        # NORMALIZATION:
        NF = np.sum(alpha_prob[:,T])        # normalization_factor
        new_A = new_A/(NF + 1e-10)          # adding a small constant to avoid division by zero

        # standardization - rows must sum up to 1:
        new_A[0,:] = new_A[0,:]/np.sum(new_A[0,:])
        new_A[1,:] = new_A[1,:]/np.sum(new_A[1,:])
        new_B[0,:] = new_B[0,:]/np.sum(new_B[0,:])
        new_B[1,:] = new_B[1,:]/np.sum(new_B[1,:])
        
        # distance between current and previous estimates (simple Euclidean distance):
        d = np.sqrt(np.sum(np.power(old_A-new_A,2)) + np.sum(np.power(old_B-new_B,2)))
        
        # stop condition:
        if d > delta:
            old_A = np.copy(new_A)
            old_B = np.copy(new_B)
        else:
            
            break
    
    return new_A, new_B, iteration, d  
        

# new_A, new_B, iteration, d = BaumWelch_algorithm(init_prob,init_A,B,sequence,delta)
# print(f'\nNew transition matrix:\n{pd.DataFrame(new_A)}\n\nNew emission probabilities:\n{pd.DataFrame(new_B)}\n\nno of iterations: {iteration}\n\nending error: {d}')

