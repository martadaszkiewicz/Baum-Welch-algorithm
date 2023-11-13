import numpy as np

# # Implementation of the backward algorithm for a Hidden Markov Model (HMM) with two states.

# # exemplary data: 
# A = np.array([[0.53, 0.47], [0.56, 0.44]], dtype=np.float64)                                    # Transition probabilities between states
# sequence = np.array([0,0,1,3,0,3,0,2]) # RRTERERS                                                 The observed sequence.
# # [R T S E]:
# emit_prob_s = np.array([[0.36, 0.19, 0.22, 0.23],[0.12, 0.44, 0.17, 0.27]],dtype=np.float64)    # Emission probabilities for each state and symbol.

# BACKWARD ALGORITHM:
def backward_algorithm(A,sequence,emit_prob_s):
    N = len(sequence)
    beta_prob = np.zeros((2, N))
    
    # initialization of the beta probabilities:     init_beta = [1, 1]:
    beta_prob[:, N-1] = 1
    
    #  iterating backward through the sequence, 
    # updating the beta probabilities based on the emission probabilities and transitions from the current state to the next states
    for j in range((N-2), -1, -1):
        beta_prob[0,j] = beta_prob[0,j+1]*A[0,0]*emit_prob_s[0,sequence[j+1]] + beta_prob[1,j+1]*A[0,1]*emit_prob_s[1,sequence[j+1]]
        beta_prob[1,j] = beta_prob[1,j+1]*A[1,1]*emit_prob_s[1,sequence[j+1]] + beta_prob[0,j+1]*A[1,0]*emit_prob_s[0,sequence[j+1]]
    
    return beta_prob # represents the probabilities of transitioning from each state to the next states at each position in the sequence.
    


# beta_prob = backward_algorithm(A,sequence,emit_prob_s)