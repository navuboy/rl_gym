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

ENV_NAME = 'Cartpole_v0'
EPISODES = 100000
TEST = 10


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
    


def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def policy_gradient():
    with tf.variable_scope("policy"):
        params = tf.get_variable("policy_parameters",[4,2])
        state = tf.placeholder("float",[None,4])
        actions = tf.placeholder("float",[None,2])
        advantages = tf.placeholder("float",[None,1])
        linear = tf.matmul(state,params)
        probabilities = tf.nn.softmax(linear)
        good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions),reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return probabilities, state, actions, advantages, optimizer

def value_gradient():
    with tf.variable_scope("value"):
        state = tf.placeholder("float",[None,4])
        newvals = tf.placeholder("float",[None,1])
        w1 = tf.get_variable("w1",[4,10])
        b1 = tf.get_variable("b1",[10])
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable("w2",[10,1])
        b2 = tf.get_variable("b2",[1])
        calculated = tf.matmul(h1,w2) + b2
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, loss


def run_episode(policy_grad, value_grad, sess):
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    reset()
    observation = robot_state.robot_state
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []


    for _ in range(20000):

        # calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_calculated,feed_dict={pl_state: obs_vector})
        action = 0 if random.uniform(0,1) < probs[0][0] else 1
        # record the transition
        states.append(observation)
        # print("angle: ", observation[2]*180/3.14)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)
        # take the action in the environment
        old_observation = observation
        reward, done = take_action(action)
        observation = robot_state.robot_state
        transitions.append((old_observation, action, reward))
        totalreward += reward

        if done:
            robot_state.done = False
            break
    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        # calculate discounted monte-carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in range(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.97
        obs_vector = np.expand_dims(obs, axis=0)
        currentval = sess.run(vl_calculated,feed_dict={vl_state: obs_vector})[0][0]

        # advantage: how much better was this action than normal
        advantages.append(future_reward - currentval)

        # update the value function towards new return
        update_vals.append(future_reward)

    # update value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})
    # real_vl_loss = sess.run(vl_loss, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})

    return totalreward


def main():

    listener()
    # outdir = '/home/imitaion/catkin_ws/src/cartpole_controller/src'
    # plotter = LivePlot(outdir)

    policy_grad = policy_gradient()
    value_grad = value_gradient()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    list_rewards = []

    for i in range(200000):
        print("Episode ", i)
        reward = run_episode(policy_grad, value_grad, sess)
        print "reward", reward
        list_rewards.append(reward)
        if(i%20==0):
            # plotter.plot(list_rewards)
            print "20 episodes done."
   
        time.sleep(0.05)
    t = 0
    for _ in range(1000):
        reward = run_episode(policy_grad, value_grad, sess)
        t += reward
    print (t / 1000)
    #env.monitor.close()
    print ("END!")


if __name__ == '__main__':
    main()
