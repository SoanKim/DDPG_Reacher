# DDPG_Reacher
This project was provided by [the Deep Reinforcement Learning Nanodegree Course of Udacity.](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893)

# Environment
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
With foundational background in policy gradient methods, it implemented a variation of the methodology known as Deep Deterministic Policy Gradient (DDPG).

The project environment is similar to, but not identical Reacher environment on the Unity. For this project, We will install an amended version of the `python/` folder from the [ML-Agents repository](https://github.com/Unity-Technologies/ml-agents).  It has been edited to include a few additional pip packages needed for the Deep Reinforcement Learning Nanodegree program.<br/>

![Alt Text](https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)

# Requirements
tensorflow==1.7.1
Pillow>=4.2.1
matplotlib
numpy>=1.11.0
jupyter
pytest>=3.2.2
docopt
pyyaml
protobuf==3.5.2
grpcio==1.11.0
torch==0.4.0
pandas
scipy
ipykernel

# The Repository Structure
* ``DDPG_Reacher.ipynb`` - Includes the environment, agent, model, and DQN functions.<br/>
* ``Report.md`` - Includes hyperparameters.<br/>
* ``result.png`` - Shows the cumulative rewards after training.<br/>
* ``actor_checkpoint.pth`` - Contains the parameters of the loca network of the actor.<br/>
* ``critic_checkpoint.pth`` - Contains the parameters of the loca network of the critic.<br/>

# Ideas for Future Work
* Hyperparameters would be fine-tuned.
* The accumulated reward decreased after 270th episode, which should be investigated. 
* [The Actor-Dueling-Critic Method for Reinforcement Learning](https://europepmc.org/article/pmc/6479875#B19-sensors-19-01547) would be implemented. In the continuous action space, we cannot output the estimation of each possible action’s advantage value. To do so, we should manually divide the action space and estimate the advantage of the action interval in each state. Through this change, the agent could learn which action interval is good when facing a specific state and pick the action belong to this interval.
