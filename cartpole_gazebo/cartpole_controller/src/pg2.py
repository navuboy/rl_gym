import tensorflow as tf
import numpy as np


import roslib
import rospy
import random
import time
import math
import csv
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelConfiguration

from control_msgs.msg import JointControllerState
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.msg import LinkState
from std_msgs.msg import Float64
from std_msgs.msg import String
from sensor_msgs.msg import Joy


import threading
from scipy.interpolate import interp1d


H = 10
batch_size = 50
learning_rate = 1e-2
gamma = 0.99
D = 4 

pubCartPosition = rospy.Publisher('/stand_cart_position_controller/command', Float64, queue_size=1)
pubJointStates = rospy.Publisher('/joint_states', JointState, queue_size=1)

reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
reset_joints = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
set_link = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)

fall = 0


rospy.init_node('cartpole_control_script')
rate = rospy.Rate(120)



class RobotState(object):
    def __init__(self):
        self.cart_x = 0.0
        self.cart_x_dot = 0.0
        self.pole_theta = 0.0
        self.pole_theta_dot = 0.0
        self.robot_state = [self.cart_x, self.cart_x_dot, self.pole_theta, self.pole_theta_dot]
        
        self.data = None
        self.latest_reward = 0.0
        self.fall = 0

        self.theta_threshold = 0.20943951023
        self.x_threshold = 0.4

        self.current_vel = 0.0
        self.done = False


robot_state = RobotState()


def reset():
    rospy.wait_for_service('/gazebo/reset_world')

    try:
        reset_world()
    except (rospy.ServiceException) as e:
        print "reset_world failed!"


        # rospy.wait_for_service('/gazebo/reset_world')
    rospy.wait_for_service('/gazebo/set_model_configuration')

    try:
        #reset_proxy.call()
        # reset_world()
        reset_joints("cartpole", "robot_description", ["stand_cart", "cart_pole"], [0.0, 0.0])


    except (rospy.ServiceException) as e:
        print "/gazebo/reset_joints service call failed"

    rospy.wait_for_service('/gazebo/pause_physics')
    try:
        pause()
    except (rospy.ServiceException) as e:
        print "rospause failed!"

    # rospy.wait_for_service('/gazebo/unpause_physics')
    
    # try:
    #     unpause()
    # except (rospy.ServiceException) as e:
    #     print "/gazebo/pause_physics service call failed"

    set_robot_state()
    robot_state.current_vel = 0
    print "called reset()"





def set_robot_state():
    robot_state.robot_state = [robot_state.cart_x, robot_state.cart_x_dot, robot_state.pole_theta, robot_state.pole_theta_dot]

def take_action(action):
    rospy.wait_for_service('/gazebo/unpause_physics')
    
    try:
        unpause()
    except (rospy.ServiceException) as e:
        print "/gazebo/pause_physics service call failed"

    
    if action == 1:
        robot_state.current_vel = robot_state.current_vel + 0.05
    else:
        robot_state.current_vel = robot_state.current_vel - 0.05


    # print "publish : ", robot_state.current_vel
    pubCartPosition.publish(robot_state.current_vel)
    
    reward = 1

    # ['cart_pole', 'stand_cart']
    if robot_state.data==None:
        while robot_state.data is None:
            try:
                robot_state.data = rospy.wait_for_message('/joint_states', JointState, timeout=5)
            except:
                print "Error getting /joint_states data."
    # print "DATA : ",robot_state.data
    # print "latest_reward: ", robot_state.latest_reward

    # if len(robot_state.data.velocity) > 0:
    #     robot_state.cart_x_dot = robot_state.data.velocity[1]
    #     robot_state.pole_theta_dot = robot_state.data.velocity[0]
    # else:
    #     robot_state.cart_x_dot = 0.0
    #     robot_state.pole_theta_dot = 0.0

    # robot_state.cart_x = robot_state.data.position[1]
    # robot_state.pole_theta = robot_state.data.position[0]
    

    set_robot_state()

    if robot_state.cart_x < -robot_state.x_threshold or robot_state.cart_x > robot_state.x_threshold or robot_state.pole_theta > robot_state.theta_threshold \
    or robot_state.pole_theta < -robot_state.theta_threshold:
       
        robot_state.done = True
        reward = 1

    else:
        reward = 1

    # rate.sleep()

    return reward, robot_state.done


def callbackJointStates(data):
    if len(data.velocity) > 0:
        robot_state.cart_x_dot = data.velocity[1]
        robot_state.pole_theta_dot = data.velocity[0]
    else:
        robot_state.cart_x_dot = 0.0
        robot_state.pole_theta_dot = 0.0
    robot_state.cart_x = data.position[1]
    robot_state.pole_theta = data.position[0]

    set_robot_state()

    print "DATA :", data


def listener():
    print "listener"
    rospy.Subscriber("/joint_states", JointState, callbackJointStates)
    

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def main():

    listener()
    reset()

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

    xs,hs,dlogps,drs,ys,tfps = [],[],[],[],[],[]
    running_reward = None
    reward_sum = 0
    episode_number = 1
    total_episodes = 10000
    init = tf.initialize_all_variables()


    # Launch the graph
    with tf.Session() as sess:
        rendering = False
        sess.run(init)
        reset() # Obtain an initial observation of the environment
        observation = robot_state.robot_state

        # Reset the gradient placeholder. We will collect gradients in 
        # gradBuffer until we are ready to update our policy network. 
        gradBuffer = sess.run(tvars)
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0
        
        while episode_number <= total_episodes:
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
            reward, done = take_action(action)
            observation = robot_state.robot_state
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
                discounted_epr = discounted_epr - np.mean(discounted_epr)
                discounted_epr = discounted_epr / np.std(discounted_epr)
                
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
                    print ('Average reward for episode %f.  Total average reward %f.' % (reward_sum/batch_size, running_reward/batch_size))
                    
                    if reward_sum/batch_size > 200: 
                        print ("Task solved in",episode_number,'episodes!')
                        break
                        
                    reward_sum = 0  
                
                reset()
                observation = robot_state.robot_state
            
    print (episode_number,'Episodes completed.')


if __name__ == '__main__':
    main()
