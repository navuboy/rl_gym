'''
A frozenlake-v0 is a 4x4 grid world which looks as follows:
SFFF       (S: starting point, safe)
FHFH       (F: frozen surface, safe)
FFFH       (H: hole, fall to your doom)
HFFG       (G: goal, where the frisbee is located)
Additionally, there is a little uncertainity in the movement
of the agent.

Q Learning - A simple q Learning algorithm is employed for the task.
The q values are stored in a table and these are updated in each iteration
to converge to their optimum values.
'''

import tensorflow as tf
import gym
import numpy as np
import random
import math

env = gym.make("FrozenLake-v0")

num_episodes = 3000
gamma = 0.99
learning_rate = 0.90

# initialize the Q table
Q = np.zeros([16, 4])


for _ in range(num_episodes):
	state = env.reset()
	done = False
	while done == False:
		# chose actions with noise
		action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n)*(1./(_+1)))
		new_state, reward, done, info = env.step(action)
		env.render()

		update = reward + (gamma*np.max(Q[new_state, :])) - Q[state, action]
		Q[state,action] += learning_rate*update 

		state = new_state





