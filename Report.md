# Report
This is the second project of [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program at Udacity. The projects use rich simulation environments from [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents). 
The code is base on the pendulum modeling of [the repository of Udacity](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum).
- [Continuous Control](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control): In the second project, I train an robotic arm to reach target locations.

# DDPG Algorithm
Modifies DPG inspired by DQN, which enables to use large state and action spaces online.<br/>
Policy gradients are stochastic, but DDPG is deterministic. Deterministic policy outputs the actual action instead of a probability.<br/>
<a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\mu'(s_t)&space;=&space;\mu(s_t|\theta^\mu_t)&plus;N)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\bg_white&space;\mu'(s_t)&space;=&space;\mu(s_t|\theta^\mu_t)&plus;N)" title="\mu'(s_t) = \mu(s_t|\theta^\mu_t)+N)" /></a>
DPG Algorithm is to select an action according to a prob distribution ($\mu$ = deterministic policy)<br/>

<a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\nabla_\theta^\mu&space;J&space;\approx&space;\mathbb{E}s_t\backsim&space;\rho^\beta[\nabla_{\theta\mu}&space;Q(s,&space;a|\theta^Q)|_s=_{st},&space;_a&space;=&space;_{\mu(st|\theta^\mu)}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\bg_white&space;\nabla_\theta^\mu&space;J&space;\approx&space;\mathbb{E}s_t\backsim&space;\rho^\beta[\nabla_{\theta\mu}&space;Q(s,&space;a|\theta^Q)|_s=_{st},&space;_a&space;=&space;_{\mu(st|\theta^\mu)}" title="\nabla_\theta^\mu J \approx \mathbb{E}s_t\backsim \rho^\beta[\nabla_{\theta\mu} Q(s, a|\theta^Q)|_s=_{st}, _a = _{\mu(st|\theta^\mu)}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;=\mathbb{E}s_t\backsim&space;\rho^\beta[\nabla_{\theta\mu}&space;Q(s,&space;a|\theta^Q)|_s=_{st},&space;_a&space;=&space;_{\mu(st)}\nabla_{\theta\mu}\mu(s|\theta^\mu)|_{s=s_t}]" target="_blank"><img src="https://latex.codecogs.com/png.latex?\bg_white&space;=\mathbb{E}s_t\backsim&space;\rho^\beta[\nabla_{\theta\mu}&space;Q(s,&space;a|\theta^Q)|_s=_{st},&space;_a&space;=&space;_{\mu(st)}\nabla_{\theta\mu}\mu(s|\theta^\mu)|_{s=s_t}]" title="=\mathbb{E}s_t\backsim \rho^\beta[\nabla_{\theta\mu} Q(s, a|\theta^Q)|_s=_{st}, _a = _{\mu(st)}\nabla_{\theta\mu}\mu(s|\theta^\mu)|_{s=s_t}]" /></a>

Create a copy of actor critic network
<a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;Q'(s,a|\theta^{Q'})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bg_white&space;Q'(s,a|\theta^{Q'})" title="Q'(s,a|\theta^{Q'})" /></a> 
and <a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\mu'(s|\theta^{Q'})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bg_white&space;\mu'(s|\theta^{Q'})" title="\mu'(s|\theta^{Q'})" /></a> 
respectively (so two actors and two critics) used for calculating the target values. <br/>
<a href="https://www.codecogs.com/eqnedit.php?latex=\bg_white&space;\theta'\leftarrow\tau\theta&plus;(1-\tau)\theta'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bg_white&space;\theta'\leftarrow\tau\theta&plus;(1-\tau)\theta'" title="\theta'\leftarrow\tau\theta+(1-\tau)\theta'" /></a> 
with 
<a href="https://www.codecogs.com/eqnedit.php?latex=\tau&space;\ll1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\tau&space;\ll1" title="\tau \ll1" /></a>. This means the target values are constrained to change slowly, greatly improving the stability of learning.<br/>

# Hyperparameters
### The parameters are based on the paper (Lillicrap et al., 2015)
<pre>
BUFFER_SIZE = int(1e5)  # replay buffer size</br>
BATCH_SIZE = 128        # minibatch size</br>
GAMMA = 0.99            # discount factor</br>
TAU = 1e-3              # for soft update of target parameters</br>
LR_ACTOR = 1e-4         # learning rate of the actor(Kinma & Ba, 2014)</br>
LR_CRITIC = 1e-3        # learning rate of the critic(Kinma & Ba, 2014)</br>
WEIGHT_DECAY = 0        # L2 weight decay</br>
</pre>

# Code Implementation
DDPG_Reacher-Final.ipynb consists of four parts: ``Network``(both actor and critic), ``agent``, ``replay buffer``, and ``main``.<br/>

* ``Network(both actor and critic)`` - Q network: state-action value function. Includes local and target networks of actor and critic.<br/>
* ``agent`` - Is is the interface between the deep neural network and the environment.<br/>
* ``replay buffer`` - The network is trained off-policy with samples from a replay buffer (finite sized cache R with (s_t, a_t, r_t, s_{t+1}) to minimize correlations between examples. t each time step, the actor and critic are updated by sampling a minibatch uniformly from the buffer .<br/>
* ``main`` - Contains the parameters of the local network of the actor.<br/>

# Ideas for Future Work
* Hyperparameters would be fine-tuned.
* The reward plot seems to need more convergence.
* [The Actor-Dueling-Critic Method for Reinforcement Learning](https://europepmc.org/article/pmc/6479875#B19-sensors-19-01547) would be implemented. In the continuous action space, we cannot output the estimation of each possible action’s advantage value. To do so, we should manually divide the action space and estimate the advantage of the action interval in each state. Through this change, the agent could learn which action interval is good when facing a specific state and pick the action belong to this interval.
