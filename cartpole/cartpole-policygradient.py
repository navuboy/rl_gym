'''
This code uses Policy gradient algorithm called REINFORCE to solve
OpenAI Gym's CartPole balancing problem.

Find evaluation at : https://gym.openai.com/evaluations/eval_MkKDu7AS1mcfffi7LWeg

'''

import tensorflow as tf
import numpy as np
import gym
import math
import csv

env = gym.make("CartPole-v0")
env.reset()

writer_file = open('rewards_pg1.csv', 'wt')
writer = csv.writer(writer_file)
writer.writerow(['total_rewards_0'])

H = 10
batch_size = 50
learning_rate = 1e-2
gamma = 0.99
D = 4 

tf.reset_default_graph()

observations = tf.placeholder(tf.float32, [None, D], name="input_x")
w1 = tf.get_variable("w1", shape=[D,H], initializer=tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, w1))
w2 = tf.get_variable("w2", shape=[H,1], initializer=tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, w2)
probability = tf.nn.sigmoid(score)

tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None,1], name="input_y")
advantages = tf.placeholder(tf.float32, name="reward_signal")

loglik = tf.log(input_y*(input_y - probability) + (1 - input_y)*(input_y + probability))
loss = -tf.reduce_mean(loglik*advantages)
newGrads = tf.gradients(loss, tvars)

adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
w1grad = tf.placeholder(tf.float32,name="batch_grad1")
w2grad = tf.placeholder(tf.float32, name="batch_grad2")
batchgrads = [w1grad, w2grad]
updategrads = adam.apply_gradients(zip(batchgrads, tvars))

def discount_rewards(r):
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(xrange(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.initialize_all_variables()
#env.monitor.start('cartpole-policygradient-monitor/', force=True)

# Launch the graph
with tf.Session() as sess:
    rendering = False
    sess.run(init)
    observation = env.reset() # Obtain an initial observation of the environment

    # Reset the gradient placeholder. We will collect gradients in 
    # gradBuffer until we are ready to update our policy network. 
    gradBuffer = sess.run(tvars)
    for ix,grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:
       	
        # if reward_sum/batch_size > 100 or rendering == True : 
        #     env.render()
        #     rendering = True
        
        # Make sure the observation is in a shape the network can handle.
        x = np.reshape(observation,[1,D])
        
        # Run the policy network and get an action to take. 
        tfprob = sess.run(probability,feed_dict={observations: x})

        action = 1 if np.random.uniform() < tfprob else 0
        #print ("ACTION :",action)
        xs.append(x) # observation
        y = 1 if action == 0 else 0 
        ys.append(y)

        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)

        reward_sum += reward

        drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

        if done: 
            episode_number += 1
            # stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            tfp = tfps
            xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[] # reset array memory

            # compute the discounted reward backwards through time
            discounted_epr = discount_rewards(epr)
            # size the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            
            # Get the gradient for this episode, and save it in the gradBuffer
            tGrad = sess.run(newGrads,feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix,grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            # If we have completed enough episodes, then update the policy network with our gradients.
            if episode_number % batch_size == 0:
                sess.run(updategrads,feed_dict={w1grad: gradBuffer[0],w2grad:gradBuffer[1]})
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                
                # Give a summary of how well our network is doing for each batch of episodes.
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
                print ('Average reward for episode %f is :%f.  Total average reward %f.' % (episode_number, reward_sum/batch_size, running_reward/batch_size))
                writer.writerow([reward_sum/batch_size])

                if reward_sum/batch_size > 200: 
                    print ("Task solved in",episode_number,'episodes!')
                    break
                    
                reward_sum = 0
            
            observation = env.reset()
        
print (episode_number,'Episodes completed.')
#env.monitor.close()

