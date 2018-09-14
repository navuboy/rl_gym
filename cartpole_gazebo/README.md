## 3D CartPole in Gazebo environment.

### Dependencies:
- Ubuntu 16.04 (http://releases.ubuntu.com/16.04/)
- ROS Kinetic (http://wiki.ros.org/kinetic)
- Gazebo 7 (http://gazebosim.org/)
- TensorFlow: 1.1.0 (https://www.tensorflow.org/) [with GPU support] 
- gym: 0.9.3 (https://github.com/openai/gym)
- Python 3.6

### File setup:
- ***cartpole_gazebo*** contains the robot model(both **.stl** files & **.urdf** file) and also the gazebo launch file -      (**cartpole_gazebo.launch**)

- ***cartpole_controller*** contains the reinforcement learning implementation of ****Policy Gradient algorithm**** for custom cartpole - (**Note: run pg.py**)

Policy Gradient for custom designed cartpole model in gazebo environment.
<p align= "center">
  <img src="/images/pg2.gif/">
</p>

 
