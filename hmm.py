def forward(states, T, observations, O):
    n = len(observations)
    prob = [{} for _ in range(n)]

    start = {1:1/3, 2:1/3, 3:1/3}

    # initialize
    for s in states:
        prob[0][s] = start[s] * O[s][observations[0]]

    for t in range(1, n):
        for s in states:
            prob[t][s] = sum(prob[t-1][prev_s] * T[prev_s][s] for prev_s in states) * O[s][observations[t]]
        # normalize
        prob_sum = sum(prob[t].values())
        for s in states:
            prob[t][s] /= prob_sum            
            
    # return prob of being in state s given observations up to t      
    return prob[-1]

def backward(states, T, observations, O):
    n = len(observations)
    prob = [{} for _ in range(n)]

    # initialize
    for s in states:
        prob[n-1][s] = 1
    
    for t in range(len(observations)-2, -1, -1):
        for s in states:
             prob[t][s] = sum(prob[t+1][next_s] * T[s][next_s] * O[s][observations[t+1]] for next_s in states)
        # normalize
        prob_sum = sum(prob[t].values())
        for s in states:
            prob[t][s]/=prob_sum   

    # return prob of being in state s given observations after t
    return prob[0]

def forward_backward(states, T, observations, O):
    return forward(states, T, observations, O) * backward(states, T, observations, O)
    


states = [1, 2, 3]
T = {
    1: {1: 1/4, 2: 1/2, 3: 1/4},
    2: {1: 1/4, 2: 1/4, 3: 1/2},
    3: {1: 1/2, 2: 1/4, 3: 1/4}
}
observations = ['r', 'g', 'b']
O = {
    1: {'r': 1/3, 'g': 1/3, 'b': 1/3},
    2: {'r': 1/3, 'g': 1/3, 'b': 1/3},
    3: {'r': 1/2, 'g': 0, 'b': 1/2}
}

print(forward(states, T, observations, O))
print(backward(states, T, observations, O))