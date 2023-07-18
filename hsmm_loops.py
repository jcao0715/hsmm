import math
import matplotlib.pyplot as plt
import numpy as np

states = np.array([0, 1, 2])
Transition = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [1, 0, 0]])
obs = np.array(['a', 'a', 'a', 'b', 'a', 'a'])
Observation = np.array([[1/3, 1, 1/3],
                        [1/3, 0, 1/3],
                        [1/3, 0, 1/3]])
Duration = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])

def q_matrix(obs, Observation):
    obs_idx = np.searchsorted(['a', 'b', 'c'], obs)
    return Observation[:, obs_idx]

q = q_matrix(obs, Observation)

def forward(states, Transition, obs, q, Duration):
    T = len(obs)
    N = len(states)
    D = len(Duration[0])
    alpha = [[0] * N for _ in range(T)]
    start = np.array([1/3, 1/3, 1/3])

    # initialization
    for i in states:
        alpha[0][i] += start[i] * Duration[i][0] * q[i][0]

    for i in states:
        alpha[1][i] += start[i] * Duration[i][1] * math.prod(q[i][s] for s in range(2))
        for j in states:
            if j!=i:
                alpha[1][i] += alpha[0][j] * Transition[j][i] * Duration[i][0] * q[i][1] 
    
    for i in states:
        alpha[2][i] += start[i] * Duration[i][2] * math.prod(q[i][s] for s in range(3))
        for d in range(2):
            for j in states:
                if j!=i:
                    alpha[2][i] += alpha[1-d][j] * Transition[j][i] * Duration[i][d] * math.prod(q[i][s] for s in range(2-d, 3))    

    # fill alpha[D:]
    for t in range(D, T):
        for j in states:
            for i in states:
                for d in range(D):
                    alpha[t][j] += alpha[t-d-1][i] * Transition[i][j] * Duration[j][d] * math.prod(q[j][s] for s in range(t-d, t+1))
    
    return alpha

fwd = forward(states, Transition, obs, q, Duration)

def backward(states, Transition, obs, q, Duration):
    T = len(obs)
    N = len(states)
    D = len(Duration[0])
    beta = [[0] * N for _ in range(T)]

    # initialization, beta[-1]
    for i in states:
        beta[T-1][i] = 1
    
    # fill beta[:-1]
    for t in range(T - 2, -1, -1):
        for i in states:
            for j in states:
                for d in range(D):
                    if t + d <= T-2:
                        beta[t][i] += beta[t + d + 1][j] * Transition[i][j] * Duration[j][d] * math.prod(q[j][s] for s in range(t+1, t + d+2))

    return beta

bwd = backward(states, Transition, obs, q, Duration)

def fb_alg(states, Transition, obs, q, Duration):
    T = len(obs)
    alpha = forward(states, Transition, obs, q, Duration)
    beta = backward(states, Transition, obs, q, Duration)
    fb_probs = []
    for t in range(len(alpha)):
        fb_probs.append([alpha[t][i] * beta[t][i] for i in states])
    
    # normalize
    fb_sum = sum(alpha[T-1][i] for i in states)
    for t in range(len(alpha)):
        for s in states:
            fb_probs[t][s] /= fb_sum if fb_sum != 0 else 1

    return fb_probs

fb_probs = fb_alg(states, Transition, obs, q, Duration)
print(fb_probs)

# plot
for state in states:
    probs = [fb_probs[t][state] for t in range(len(fb_probs))]
    plt.plot(range(1, len(fb_probs) + 1), probs, label=f'State {state}')

plt.xlabel('Time step')
plt.ylabel('Probability')
plt.title('Probabilities for Each State')
plt.legend()
plt.show()