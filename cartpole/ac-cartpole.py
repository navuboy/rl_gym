import tensorflow as tf
import numpy as np
import gym
import csv
import copy
import random

writer_file = open('rewards_ac_cartpole_2.csv', 'wt')
writer = csv.writer(writer_file)
writer.writerow(['total_rewards'])

EPISODES = 10000
env = gym.make('CartPole-v0')
# env.seed(1)
# env = env.unwrapped

# create network graph
D = env.observation_space.shape[0]
A = env.action_space.n
H = 10
actor_learning_rate = 0.001
critc_learning_rate = 0.01
gamma = 0.95
render = False
memory_size = 200
batch_size = 64

tf.reset_default_graph()
input_x = tf.placeholder(tf.float32, [None, D], name="input_x")
true_q = tf.placeholder(tf.float32,  name = "true_q")

################################## Critic Network ###################################
W1 = tf.get_variable("W1", shape=[D, H],
           initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(input_x,W1))

W2 = tf.get_variable("W2", shape=[H, 1],
           initializer=tf.contrib.layers.xavier_initializer())

critic_fc3 = tf.matmul(layer1, W2)
## ----------------------------------------------------------------------------------
t_W1 = tf.get_variable("t_W1", shape=[D, H],
           initializer=tf.contrib.layers.xavier_initializer())
t_layer1 = tf.nn.relu(tf.matmul(input_x, t_W1))

t_W2 = tf.get_variable("t_W2", shape=[H, 1],
           initializer=tf.contrib.layers.xavier_initializer())

critic_t_fc3 = tf.matmul(t_layer1, t_W2)

diffs = critic_fc3 - true_q 
critic_loss = -tf.reduce_mean(tf.square(diffs))

critic_optimizer = tf.train.AdamOptimizer(learning_rate=critc_learning_rate).minimize(critic_loss)


##################################### Actor Network #################################
actor_fc1 = tf.contrib.layers.fully_connected(inputs = input_x,\
	num_outputs = H,\
	activation_fn= tf.nn.relu,\
	weights_initializer=tf.contrib.layers.xavier_initializer())
actor_fc2 = tf.contrib.layers.fully_connected(inputs = actor_fc1,\
	num_outputs = A,\
	activation_fn= tf.nn.relu,\
	weights_initializer=tf.contrib.layers.xavier_initializer())
actor_fc3 = tf.contrib.layers.fully_connected(inputs = actor_fc2,\
	num_outputs = A,\
	activation_fn= None,\
	weights_initializer=tf.contrib.layers.xavier_initializer())

output = tf.nn.softmax(actor_fc3)

input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
discounted_rewards = tf.placeholder(tf.float32, name="discounted_rewards")
neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(logits=actor_fc3, labels=input_y)
product = neg_log_likelihood * discounted_rewards
actor_loss = tf.reduce_mean(product) # no need for -ve sign if using tf.nn.softmax_cross_entr..... 
							   		 # as it gives neg_log_likelihood

actor_optimizer = tf.train.AdamOptimizer(learning_rate=actor_learning_rate).minimize(actor_loss)


init = tf.initialize_all_variables()

def discounted_rewards_(r):
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(xrange(len(r))):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add

	return discounted_r

def choose_action(out):
	action = np.random.choice(range(out.shape[1]), p=out.ravel())
	return action


with tf.Session() as sess:
	sess.run(init)

	xs, drs, ys = [], [], []
	
	episode_number = 0
	reward_sum = 0
	reward_sum_buffer = []
	current_state = env.reset()
	memory_target = []
	memory_states = []
	done = False
	goal_reached = False

	while not goal_reached:
		x = np.reshape(current_state, [1, D])
		out = sess.run(output, feed_dict={input_x: x})
		action = choose_action(out)
		xs.append(x)
		temp_y = np.zeros(2)
		temp_y[action] = 1
		ys.append(temp_y)

		next_state, reward, done, _ = env.step(action)
		# if render:
		# 	env.render()
		drs.append(reward)
		reward_sum += reward

		if not done:
			q_pred = sess.run(critic_t_fc3, feed_dict={input_x: np.reshape(next_state,[1, D])})
			update = reward + gamma*q_pred
		else:
			update = reward

		memory_target.append(update)
		memory_states.append(x)

		# if episode ends, find discounted rewards and 
		# find gradients for the episode
		
		if done:

			episode_number += 1
			epx = np.vstack(np.array(xs))
			epy = np.vstack(np.array(ys))
			epr = np.vstack(np.array(drs))
 
			discounted_rs =  discounted_rewards_(drs)
		
			xs, ys, drs = [], [], []


			memory_states_temp = copy.copy(memory_states)
			memory_targets_temp = copy.copy(memory_target)

			memory_states_temp = np.vstack(memory_states_temp)
			memory_targets_temp = np.vstack(memory_targets_temp)

			temp_list = zip(memory_states_temp, memory_targets_temp)
			random.shuffle(temp_list)
			ep_states, ep_targets = zip(*temp_list[:batch_size])

			
			q_pred_ = sess.run(critic_t_fc3, feed_dict={input_x: epx})
			sess.run(actor_optimizer,  feed_dict={discounted_rewards: (q_pred_- np.mean(discounted_rs)), input_x: epx, input_y: epy})
			sess.run(critic_optimizer, feed_dict={true_q: ep_targets, input_x: ep_states})
			reward_sum_buffer.append(reward_sum)
	
			if episode_number % 100 == 0:
				average_per_100_eps = sum(reward_sum_buffer)/100
				if average_per_100_eps == 200.00: # acieved the goal.
					goal_reached = True 
				t_W1 = tf.identity(W1)
				t_W2 = tf.identity(W2)	
				print "Average reward for ", episode_number," episodes is :", average_per_100_eps
				reward_sum_buffer = []
				writer.writerow([average_per_100_eps])
			if episode_number % memory_size == 0:
				memory_states = []
				memory_target = []

			if reward_sum == 200.0:
				render = True
			reward_sum = 0
			current_state = env.reset()
		
		current_state = next_state



			
			










