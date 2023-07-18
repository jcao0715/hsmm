import math
import matplotlib.pyplot as plt

states = [1, 2, 3]
Transition = {
    1: {1: 0, 2: 1, 3: 0},
    2: {1: 0, 2: 0, 3: 1},
    3: {1: 1, 2: 0, 3: 0},
}
obs = {1: 'a', 2: 'a', 3: 'a', 4: 'b', 5: 'a', 6: 'a'}
Observation = {
    1: {'a': 1/3, 'b': 1, 'c': 1/3},
    2: {'a': 1/3, 'b': 0, 'c': 1/3},
    3: {'a': 1/3, 'b': 0, 'c': 1/3},
}
Duration = {
    1: {1: 1, 2: 0, 3: 0},
    2: {1: 0, 2: 1, 3: 0},
    3: {1: 0, 2: 0, 3: 1},
}

def q_matrix(obs, Observation):
    return {state: {i: Observation[state][obs[i]] for i in obs.keys()} for state in Observation}

q = q_matrix(obs, Observation)

def forward(states, Transition, obs, q, Duration):
    T = len(obs)
    N = len(states)
    D = len(Duration[1])
    alpha = {timestep:{} for timestep in range(1, T+1)}
    start = {1: 1/3, 2: 1/3, 3: 1/3}

    # initialization
    # alpha[1]
    for i in states:
        alpha[1][i] = start[i] * Duration[i][1] * q[i][1]
    # alpha[2]
    for i in states:
        alpha[2][i] = start[i] * Duration[i][2] * math.prod(q[i][s] for s in range(1, 3))
        for j in states:
            if j != i:
                alpha[2][i] += alpha[1][j] * Transition[j][i] * Duration[i][1] * q[i][2]
    # alpha[3]
    for i in states:
        alpha[3][i] = start[i] * Duration[i][3] * math.prod(q[i][s] for s in range(1, 4))
        for d in range(1, 3):
            for j in states:
                if j != i:
                    alpha[3][i] += alpha[3-d][j] * Transition[j][i] * Duration[i][d] * math.prod(q[i][s] for s in range(4-d, 4))
    # fill alpha[4:]
    for t in range(D+1, T+1):
        for j in states:
            alpha[t][j] = 0
            for i in states:
                for d in range(1, D+1):
                    alpha[t][j] += alpha[t-d][i] * Transition[i][j] * Duration[j][d] * math.prod(q[j][s] for s in range(t-d+1, t+1))
                
    return alpha

fwd = forward(states, Transition, obs, q, Duration)

def backward(states, Transition, obs, q, Duration):
    T = len(obs)
    N = len(states)
    D = len(Duration[1])
    beta = {timestep: {} for timestep in range(1, T + 1)}

    # initialization, fill beta[-1]
    for i in states:
        beta[T][i] = 1

    # fill beta[:-1]
    for t in range(T - 1, 0, -1):
        for i in states:
            beta[t][i] = 0
            for j in states:
                for d in range(1, D + 1):
                    if t + d <= T:
                        beta[t][i] += beta[t + d][j] * Transition[i][j] * Duration[j][d] * math.prod(q[j][s] for s in range(t + 1, t + d + 1))

    return beta

bwd = backward(states, Transition, obs, q, Duration)

def fb_alg(states, Transition, obs, q, Duration):
    T = len(obs)
    alpha = forward(states, Transition, obs, q, Duration)
    beta = backward(states, Transition, obs, q, Duration)
    fb_probs = []
    for t in range(1, len(alpha)+1):
        fb_probs.append({i: alpha[t][i] * beta[t][i] for i in states})
    # normalize
    fb_sum = sum(alpha[T][i] for i in states)
    for t in range(len(alpha)):
        for s in states:
            fb_probs[t][s] /= fb_sum if fb_sum != 0 else 1
    return fb_probs

fb_probs = fb_alg(states, Transition, obs, q, Duration)

# plot each state
for state in states:
    probs = [fb_probs[t][state] for t in range(len(fb_probs))]
    plt.plot(range(1, len(fb_probs) + 1), probs, label=f'State {state}')

# label
plt.xlabel('Time step')
plt.ylabel('Probability')
plt.title('Probabilities for Each State')
plt.legend()
plt.show()