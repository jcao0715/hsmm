import matplotlib.pyplot as plt
import numpy as np

states = np.array([0, 1, 2])
Transition = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [1, 0, 0]])
obs = np.array(['a', 'b', 'c', 'b', 'a', 'c'])
Observation = np.array([[1/2, 1/4, 1/5],
                        [1/4, 1/4, 1/5],
                        [1/4, 1/2, 3/5]])
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
    alpha[0] = start * Duration[:, 0] * q[:, 0]
    
    alpha[1] = start * Duration[:, 1] * np.prod(q[:, :2], axis=1)
    alpha[1] += np.sum(alpha[0] * Transition * Duration * q[:, 1], axis=1)

    alpha[2] = start * Duration[:, 2] * np.prod(q[:, :3], axis=1)
    for d in range(2):
        alpha[2] += alpha[1 - d] @ Transition * Duration[:, d] * np.prod(q[:, 2 - d:3], axis=1)

    # fill alpha[D:]
    for t in range(D, T):
        for d in range(D):
            alpha[t] += alpha[t - d - 1] @ Transition * Duration[:, d] * np.prod(q[:, t-d:t+1], axis=1)

    return alpha

fwd = forward(states, Transition, obs, q, Duration)

# plot
for state in states:
    probs = [fwd[t][state] for t in range(len(obs))]
    plt.plot(range(len(obs)), probs, label =f"State {state}")

plt.xlabel('Time step')
plt.ylabel('Probability of state')
plt.title('Forward algorithm')
plt.legend()
plt.show()

def backward(states, Transition, obs, q, Duration):
    T = len(obs)
    N = len(states)
    D = len(Duration[0])
    beta = [[0] * N for _ in range(T)]

    # initialization, beta[-1]
    beta[-1] = np.ones(N)
    
    # fill beta[:-1]
    for t in range(T - 2, -1, -1):
        for d in range(D):
            if t + d <= T-2:
                beta[t] += np.sum(beta[t+d+1] * Transition * Duration[:, d] * np.prod(q[:, t+1:t+d+2], axis=1), axis=1)
    return beta

bwd = backward(states, Transition, obs, q, Duration)

def fb_alg(states, Transition, obs, q, Duration):
    T = len(obs)
    alpha = forward(states, Transition, obs, q, Duration)
    beta = backward(states, Transition, obs, q, Duration)
    fb_probs = np.array(alpha) * np.array(beta)
    
    # normalize
    fb_sum = np.sum(alpha[T-1])
    fb_probs /= fb_sum if fb_sum != 0 else 1
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