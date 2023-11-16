import numpy as np

# # Implementation of the forward algorithm for a Hidden Markov Model (HMM) with two states.

# # exemplary data: 
# init_prob = np.array([0.65, 0.35], dtype=np.float64)                                            # Initial probabilities of being in each state
# A = np.array([[0.53, 0.47], [0.56, 0.44]], dtype=np.float64)                                    # Transition probabilities between states
# sequence = np.array([0,0,1,3,0,3,0,2]) # RRTERERS                                                 The observed sequence.
# # [R T S E]:
# emit_prob_s = np.array([[0.36, 0.19, 0.22, 0.23],[0.12, 0.44, 0.17, 0.27]],dtype=np.float64)    # Emission probabilities for each state and symbol.

# FORWARD ALGORITHM:
def forward_algorithm(init_prob,A,sequence,emit_prob_s):
    N = len(sequence)
    alpha_prob = np.zeros((2,N))
    
    # initialization of the alpha probabilities for each state:
    current_alphas = np.zeros(2)
    current_alphas[0] = init_prob[0]*emit_prob_s[0,sequence[0]]
    current_alphas[1] = init_prob[1]*emit_prob_s[1,sequence[0]]
    
    alpha_prob[0,0] = current_alphas[0]
    alpha_prob[1,0] = current_alphas[1]

    # iterating through the sequence, 
    # updating the alpha probabilities based on the emission probabilities and transitions from the previous state to the current state
    for i in range(1,N):
        alpha_s1 = current_alphas[0]
        alpha_s2 = current_alphas[1]
        current_alphas[0] = emit_prob_s[0,sequence[i]]*(alpha_s1*A[0,0]+alpha_s2*A[1,0])
        current_alphas[1] = emit_prob_s[1,sequence[i]]*(alpha_s2*A[1,1]+alpha_s1*A[0,1])
        alpha_prob[0,i] = current_alphas[0]
        alpha_prob[1,i] = current_alphas[1]
    
    return alpha_prob # represents the probabilities of being in each state at each position in the sequence

# alpha_prob = forward_algorithm(init_prob,A,sequence,emit_prob_s)
