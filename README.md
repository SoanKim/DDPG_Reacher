### DDPG_Reacher
This project was provided by [the Deep Reinforcement Learning Nanodegree Course of Udacity.](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) The code is base on the pendulum modeling of [the repository of Udacity](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum).
### Environment
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
With foundational background in policy gradient methods, it implemented a variation of the methodology known as Deep Deterministic Policy Gradient (DDPG).

The project environment is similar to, but not identical Reacher environment on the Unity. For this project, We will install an amended version of the `python/` folder from the [ML-Agents repository](https://github.com/Unity-Technologies/ml-agents).  It has been edited to include a few additional pip packages needed for the Deep Reinforcement Learning Nanodegree program.<br/>

![Alt Text](https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif)

# Dependencies 

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.
    - **Linux** or **Mac**:
    
    `conda create --name drlnd python=3.6
    source activate drlnd`
    
    - **Windows**:
    
    `conda create --name drlnd python=3.6 
    activate drlnd`
    
2. If running in **Windows**, ensure you have the "Build Tools for Visual Studio 2019" installed from this [site](https://visualstudio.microsoft.com/downloads/). This [article](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30) may also be very helpful. This was confirmed to work in Windows 10 Home.
3. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.
    - Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
    - Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
4. Clone the repository (if you haven't already!), and navigate to the `python/` folder. Then, install several dependencies.
    
    `git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .`
    
5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.
    
    `python -m ipykernel install --user --name drlnd --display-name "drlnd"`
    
6. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu.

### **Getting Started**

1. Download the environment from one of the links below. You need only select the environment that matches your operating system:
    - ***Version 1: One (1) Agent***
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
    - ***Version 2: Twenty (20) Agents***
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (*For Windows users*) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.
    
    (*For AWS*) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment. You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (*To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.*)
    
2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file.

### Requirements
tensorflow==1.7.1<br/>
Pillow>=4.2.1<br/>
matplotlib<br/>
numpy>=1.11.0<br/>
jupyter<br/>
pytest>=3.2.2<br/>
docopt<br/>
pyyaml<br/>
protobuf==3.5.2<br/>
grpcio==1.11.0<br/>
torch==0.4.0<br/>
pandas<br/>
scipy<br/>
ipykernel<br/>

### The Repository Structure
* ``DDPG_Reacher-Final.ipynb`` - Includes the environment, agent, model, and DQN functions.<br/>
* ``Report.md`` - Includes hyperparameters, code implementation and future work.<br/>
* ``result.png`` - Shows the cumulative rewards after training.<br/>
* ``actor_checkpoint.pth`` - Contains the parameters of the local network of the actor.<br/>
* ``critic_checkpoint.pth`` - Contains the parameters of the local network of the critic.<br/>
