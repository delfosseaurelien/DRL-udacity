# DRL-udacity

This github includes all the codes for the intermediate and final project of the deep reinforcement learning course.

### DQN_PROJECT:

The task is to create an agent that is able to collect bananas. A reward of +1 is provided for yellow banana and -1 for blue banana. The objectif is to collect 13 yellow bananas successively. The state space contains the agent velocity.

The agent have to choose four discrete actions (move forward,move backward, turn left, turn right). The Benchmarch implementation solves the environement in 1800 episodes.

### CONTINUOUS_PROJECT:

The task is to create an agent that is able to move to a target location controlling a double jointed arm. A reward of +0.1 is provided for each step that the agent's hand is in the goal location.
The objectif is to maintain position at the target location for as many time steps as possible.

The agent have to predict a vector of four continuous actions between [-1,1] which correspond to the force on the arm.
The observations space is a vector of 33 variables corresponding to position, rotation, velocity and angular velocities.

We choose to work with the first version of the environnment with 1 agent. The task is to get an average score of +30 over 100 consecutive episodes. To do this, we choose to to work with a deep deterministic policy gradient.

### MULTIAGENTS PROJECT

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).


# Dependencies
To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
 
2. Download the environement at this adres for linux
  * DQN_PROJECT : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
  * CONTINUOUS_PROJECT :  https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip
  * MULTIAGENTS_PROJECT : https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip

3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/delfosseaurelien/DRL-udacity.git
cd DRL-udacity
pip install -r requirements.txt
```


4. In each subfolder of project, run he file main.py
```bash
python main.py
```

