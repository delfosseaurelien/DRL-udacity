# Introduction 

The task is to create an agent that is able to collect bananas. A reward of +1 is provided for yellow banana and -1 for blue banana.
The objectif is to collect 13 yellow bananas successively. The state space contains the agent velocity.

The agent have to choose four discrete actions (move forward,move backward, turn left, turn right).
The Benchmarch implementation solves the environement in 1800 episodes.

# Dependencies

You need to install the environnement, put it at the root of the folder and call like this:

env = UnityEnvironment(file_name="/Banana_Linux_NoVis/Banana.x86_64")

You can download the environement for linux here:
https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip

# Run

You just have to run the file navigation.py to launch the training of the agent.
You need to specify in the file when building the agent if you want to launch the training with the duelling network or not.

## Implementation
### Double DQN

A double q network is implemented here.
Network architecture:
  * Fully connected 64
  * Fully connected 64
  * Output actions space
  
### Double DQN + Duelling network

The duelling architecture converges in 504 episodes with gamma = 0.9.
We update softly the parameters of the target network.
We update de target network every 4 steps.
Duelling architecture : 

* Fully connected 64
* Fully connected 64

* Fully connected 64 Advantage
* Fully connected 64 Value

* Fully connected 64 Advantage
* Fully connected 1 Value
* Sum Advantage + value - Mean_advantage

* Output actions space
  
# Improvement

The main improvement is to implement the prioritized experience replay. It allows the agent to select the experience with a high priority. The priority is calculated with the temporal difference error and save in a binary tree to accelerate the access to the data. The priority in the tree is updated each time the target is calculated.
(Impementation coming soon).
