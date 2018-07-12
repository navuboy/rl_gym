'''
Breakout-v0 using Full Deep Q Learning
observation dimensions (210, 160, 3)
actions ['NOOP', 'FIRE','RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']

'''
import tensorflow as tf
import gym
import numpy as np
import math
import random
from matplotlib import pyplot as plt
import itertools
import sys
from collections import deque, namedtuple

env = gym.make("Breakout-v0")

#observation = env.reset()

#print env.get_action_meanings()

#plt.figure()
#plt.imshow(env.render(mode='rgb_array'))

#[env.step(4) for x in range(1)]
#plt.figure()
#plt.imshow(env.render(mode='rgb_array'))
#plt.imshow(observation[34:-16,:,:])
#plt.imshow(observation)
#env.render(close=True)

#plt.show()

VALID_ACTIONS = [0, 1, 2, 3]

tf.reset_default_graph()

# input_preprocessor graph
input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
output = tf.image.rgb_to_grayscale(input_state)
# image, offset_height, offset_width, target_height, target_width
output = tf.image.crop_to_bounding_box(output, 34, 0, 160, 160)
output = tf.image.resize_images(output, [84, 84])
output = tf.squeeze(output)

# build estimator model
# input is 4 grayscale frames of 84, 84 each
X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name='X')
# target value
y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name='y')
# which action was chosen
actions_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="actions")

X = tf.to_float(X_pl) / 255.0
batch_size = tf.shape(X_pl)[0]

with tf.variable_scope("estimator"):
	# three convolutional layers --------------------------------------------------
	conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn=tf.nn.relu)
	conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
	conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

	# fully connected layers
	flattened = tf.contrib.layers.flatten(conv3)
	fc1 = tf.contrib.layers.fully_connected(flattened, 512)
	predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

	# get predictions for chosen actions only
	gather_indices = tf.range(batch_size)*tf.shape(predictions[1] + actions_pl)
	action_predictions = tf.gather(tf.reshape(predictions, [-1]), gather_indices)


# build target model -----------------------------------------------------------
with tf.variable_scope("target"):
	# three convolutional layers
	t_conv1 = tf.contrib.layers.conv2d(X, 32, 8, 4, activation_fn=tf.nn.relu)
	t_conv2 = tf.contrib.layers.conv2d(t_conv1, 64, 4, 2, activation_fn=tf.nn.relu)
	t_conv3 = tf.contrib.layers.conv2d(t_conv2, 64, 3, 1, activation_fn=tf.nn.relu)

	# fully connected layers
	t_flattened = tf.contrib.layers.flatten(t_conv3)
	t_fc1 = tf.contrib.layers.fully_connected(t_flattened, 512)
	t_predictions = tf.contrib.layers.fully_connected(t_fc1, len(VALID_ACTIONS))

	# get predictions for chosen actions only
	t_gather_indices = tf.range(batch_size)*tf.shape(t_predictions[1] + actions_pl)
	t_action_predictions = tf.gather(tf.reshape(t_predictions, [-1]), t_gather_indices)

# calculate loss ----------------------------------------------------------------
losses = tf.squared_difference(y_pl, action_predictions)
loss = tf.reduce_mean(losses)

# optimizer parameters
optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
train_op = optimizer.minimize(loss, global_step=tf.contrib.framework.get_global_step())

def input_preprocessor(sess, state):
	return sess.run(output, feed_dict={input_state: state})
	
def predict(sess, s, network):
	'''
	s: shape [batch_size, 4, 160, 160, 3]
	returns: shape[batch_size, NUM_VALID_ACTIONS]
	'''
	if network == "estimator":
		return sess.run(predictions, feed_dict={X_pl: s})
	else:
		return sess.run(t_predictions, feed_dict={X_pl: s})

def update(sess, s, a, y):
	'''
	s: shape [batch_size, 4, 160, 160, 3]
	a: chosen actions of shape [batch_size]
	y: targets of shape [batch_size]
	returns: calculated loss on the batch
	'''
	_, _loss = sess.run([train_op, loss], feed_dict={X_pl: s, y_pl: y, actions_pl: a})
	return _loss

def copy_model_parameters(sess):
	e_params = [t for t in tf.trainable_variables() if t.name.startswith("estimator")]
	e_params = sorted(e_params, key=lambda v: v.name)
	t_params = [t for t in tf.trainable_variables() if t.name.startswith("target")]
	t_params = sorted(t_params, key=lambda v: v.name)

	update_ops = []
	for e_v, t_v in zip(e_params, t_params):
		op = t_v.assign(e_v)
		update_ops.append(op)
	sess.run(update_ops)

def epsilon_greedy_policy(nA, sess, observation, epsilon):
	A = np.ones(nA, dtype=float)*epsilon/nA
	q_values = predict(sess,np.expand_dims(observation, 0), "estimator")[0]
	best_action = np.argmax(q_values)
	A[best_action] += (1.0 - epsilon)

	return A

def deep_q_learning(sess, 
					env, 
					num_episodes, 
					replay_memory_size=500000, 
					replay_memory_init_size=50000,
					update_target_every=10000,
					discount_factor=0.99,
					epsilon_start=1.0,
					epsilon_end=0.1,
					epsilon_decay_steps=500000,
					batch_size=32):
	
	Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
	
	# the replay memory
	replay_memory = []

	# get the current time step
	total_t = sess.run(tf.contrib.framework.get_global_step())
	# the epsilon decay schedule
	epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

	# populating the replay memory with initial experience
	state = env.reset()
	state = input_preprocessor(sess, state)
	state = np.stack([state]*4, axis=2)
	for i in range(replay_memory_init_size):
		action_probs = epsilon_greedy_policy(len(VALID_ACTIONS), sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
		action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
		next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
		next_state = input_preprocessor(sess, next_state)
		next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)
		replay_memory.append(Transition(state, action, reward, next_state, done))
		if done:
			state = env.reset()
			state = input_preprocessor(sess, state)
			state = np.stack([state]*4, axis=2)
			print "populating replay memory ... current episode: ", i

	for i_episode in range(num_episodes):

		# reset the environment
		state = env.reset()
		state = input_preprocessor(sess, state)
		state = np.stack([state]*4, axis=2)
		loss = None

		# one step in the environment
		for t in itertools.count():

			# epsilon for this time step
			epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

			# update target after regular intervals
			if total_t % update_target_every == 0:
				copy_model_parameters(sess)

			# print out which step are we on
			print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
                    t, total_t, i_episode + 1, num_episodes, loss))
			sys.stdout.flush()

			# take a step
			action_probs = epsilon_greedy_policy(len(VALID_ACTIONS), sess, state, epsilon)
			action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
			next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
			next_state = input_preprocessor(sess, next_state)
			next_state = np.append(state[:,:,1:], np.expand_dims(next_state, 2), axis=2)

			# if replay memory is full, pop the first element
			if len(replay_memory) == replay_memory_size:
				replay_memory.pop(0)

			# save the transition in replay memory
			replay_memory.append(Transition(state, action, reward, next_state, done))

			# sample a minibatch from the replay memory
			samples = random.sample(replay_memory, batch_size)
			states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

			# calculate the q values and targets
			q_values_next = predict(sess, next_states_batch, "target")
			targets_batch = reward_batch + np.invert(done_batch).astype(np.float32)*discount_factor*np.amax(q_values_next, axis=1)

			# perform gradient descent update
			states_batch = np.array(states_batch)
			loss = update(sess, states_batch, action_batch, targets_batch)

			if done:
				break

			state = next_state
			total_t += 1

# create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	deep_q_learning(sess, env, 10000)
	
	
	