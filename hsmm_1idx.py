import math

def q_matrix(obs, Observation):
    return {state: [Observation[state][o] for o in obs] for state in Observation}


def forward(states, Transition, obs, q, Duration):
    T = len(obs)
    N = len(states)
    D = len(Duration[1])
    alpha = [{} for _ in range(T)]
    start = {1: 1/3, 2: 1/3, 3: 1/3}

    # initialization, fills alpha[0]
    for j in states:
        alpha[0][j] = start[j] * Duration[j][1] * q[j][0]
        # alpha[0][j] = start[j] * q[j][0]
    # fill alpha[1:]
    # for t in range(1, T):
    #     for j in states:
    #         alpha[t][j] = 0
    #         for d in range(1, D + 1):
    #             if t-d >= 0:
    #                alpha[t][j] += sum(alpha[t-d][i] * Transition[i][j] * Duration[j][d] for i in states) * math.prod(q[j][s] for s in range(t - d + 1, t + 1))  
    for t in range(1, T):
        for j in states:
            alpha[t][j] = 0
            for i in states: # sum from i=1 to N
                for d in range(1, D + 1): # sum from d=1 to D
                    if t - d >= 0:
                        alpha[t][j] += alpha[t-d][i] * Transition[i][j] * Duration[j][d]
                        alpha[t][j] *= math.prod(q[j][s] for s in range(t - d + 1, t + 1)) # product from s=t-d+1 to t
    # # normalize
    # for t in range(T):
    #     alpha_sum = sum(alpha[t].values())
    #     for s in states:
    #         if alpha_sum == 0:
    #             break
    #         alpha[t][s] /= alpha_sum  
    return alpha


def backward(states, Transition, obs, q, Duration):
    T = len(obs)
    N = len(states)
    D = len(Duration[1])
    beta = [{} for _ in range(T)]

    # initialization, fills beta[-1]
    for i in states:
        beta[-1][i] = 1
    # fill beta[:-1]
    # for t in range(T - 2, -1, -1):
    #     for i in states:
    #         beta[t][i] = 0
    #         for d in range(1, D + 1):
    #             if t + d <= T - 1:
    #                 beta[t][i] += sum(beta[t+d][j] * Transition[i][j] * Duration[j][d] * math.prod(q[i][s] for s in range(t + 1, t + d + 1)) for j in states) 
    for t in range(T - 2, -1, -1):
        for i in states:
            beta[t][i] = 0
            for j in states: # sum from j=1 to N
                for d in range(1, D + 1): # sum from d=1 to D
                    if t + d <= T - 1:
                        beta[t][i] += beta[t+d][j] * Transition[i][j] * Duration[j][d]
                        beta[t][i] *= math.prod(q[j][s] for s in range(t + 1, t + d + 1)) # product from s=t+1 to t+d
    # # normalize
    # for t in range(T):
    #     beta_sum = sum(beta[t].values())
    #     for s in states:
    #         if beta_sum == 0:
    #             break
    #         beta[t][s] /= beta_sum
    return beta

def fb_alg(states, Transition, obs, q, Duration):
    alpha = forward(states, Transition, obs, q, Duration)
    beta = backward(states, Transition, obs, q, Duration)
    fb_probs = []
    for t in range(len(alpha)):
        # fb_probs.append({i: alpha[t][i] * beta[len(beta) - t - 1][i] for i in states})
        fb_probs.append({i: alpha[t][i] * beta[t][i] for i in states})
        # fb_sum = sum(fb_probs[t].values())
        # for s in states:
        #     fb_probs[t][s] /= fb_sum if fb_sum != 0 else 1
    # normalize
    fb_sum = sum(alpha[-1][i] for i in states)
    for t in range(len(alpha)):
        for s in states:
            fb_probs[t][s] /= fb_sum if fb_sum != 0 else 1
    return fb_probs


states = [1, 2, 3]
Transition = {
    1: {1: 0, 2: 1, 3: 0},
    2: {1: 0, 2: 0, 3: 1},
    3: {1: 1, 2: 0, 3: 0},
}
obs = ['a', 'a', 'a', 'b', 'a', 'a']
Observation = {
    1: {'a': 1/3, 'b': 1, 'c': 1/3},
    2: {'a': 1/3, 'b': 0, 'c': 1/3},
    3: {'a': 1/3, 'b': 0, 'c': 1/3},
}
q = q_matrix(obs, Observation)
Duration = {
    1: {1: 1, 2: 0, 3:0},
    2: {1: 0, 2: 1, 3: 0},
    3: {1: 0, 2: 0, 3: 1},
}

print(f"q_matrix: {q_matrix(obs, Observation)}")
print(f"forward probabilities: {forward(states, Transition, obs, q, Duration)}")
print(f"backward probabilities: {backward(states, Transition, obs, q, Duration)}")
print(f"forward backward result: {fb_alg(states, Transition, obs, q, Duration)}")