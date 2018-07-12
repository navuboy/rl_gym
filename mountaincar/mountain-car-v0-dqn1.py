'''
Environment - MountainCar-v0
actions = 0(LEFT), 1(STAY), 2(RIGHT)
state space (continuous) dimension = 2(car position, car velocity)

'''

import tensorflow as tf
import gym
import numpy as np
import math
import random
import copy

env = gym.make("MountainCar-v0")

learning_rate = 1e-2
memory_size = 100000
batch_size = 64
gamma = 0.99
epsilon_max = 1
epsilon_min = 0.1

tf.reset_default_graph()

# computation graph
observations = tf.placeholder(tf.float32, [None, 2], name="input_x")
W1 = tf.get_variable("W1", shape=[2, 64], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape=[64, 2], initializer=tf.contrib.layers.xavier_initializer())
Qpredict = tf.matmul(layer1, W2)

Qtarget = tf.placeholder(tf.float32, [None, 2], name="input_y")
error = Qtarget - Qpredict

# mean square error loss function
loss = -tf.reduce_mean(tf.square(error))
adam = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.initialize_all_variables()

# start the session
with tf.Session() as sess:
	sess.run(init)
	state_memory = []
	target_memory = []
	for  _ in range(1000):
		done = False
		state = env.reset()
		total_reward = 0
		while done == False:
			observation = np.reshape(state, [1, 2])
			state_memory.append(observation)
			_Qpredict = sess.run(Qpredict, feed_dict={observations: observation})
			epsilon = epsilon_min + (epsilon_max - epsilon_min)*(math.exp(-0.001*_))
			action_temp = 0
			# chose action using e-greedy policy
			if random.random() < epsilon:
				action = random.randint(0, 1)
				# map action 1 to 2(RIGHT) 
				if action == 1:
					action_temp = 2		
				new_state, reward, done, info = env.step(action_temp)
			else:
				action = np.argmax(_Qpredict)
				new_state, reward, done, info = env.step(action)

			total_reward += reward
			#env.render()

			_Qout = sess.run(Qpredict, feed_dict={observations: np.reshape(new_state, [1, 2])})
			_maxQout = np.max(_Qout)
			_Qtarget = _Qpredict[:]

			if done == False:
				update = reward + (gamma*_maxQout)
			else:
				update = reward

			_Qtarget[0][action] = update

			target_memory.append(_Qtarget)

			# experience replay
			sample_size = min(batch_size, len(state_memory))
			
			state_memory_temp = np.vstack(copy.copy(state_memory))
			target_memory_temp = np.vstack(copy.copy(target_memory))

			temp_list = zip(state_memory_temp, target_memory_temp)
			random.shuffle(temp_list)
			_states, _targets = zip(*temp_list)
			sess.run(adam, feed_dict={observations: _states, Qtarget: _targets})

			if state_memory >= memory_size:
				state_memory = []
				target_memory = []
		print "reward in episode ",_, " is ", total_reward
