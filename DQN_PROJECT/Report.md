#Introduction 

The task is to create an agent that is able to collect bananas. A reward of +1 is provided for yellow banana and -1 for blue banana.
The objectif is to collect 13 yellow bananas successively. The state space contains the agent velocity.

The agent have to choose four discrete actions (move forward,move backward, turn left, turn right).
The Benchmarch implementation solves the environement in 1800 episodes.

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
  
