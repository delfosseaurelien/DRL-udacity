from unityagents import UnityEnvironment
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
%matplotlib inline

from ddpg_agent import Agent

env = UnityEnvironment(file_name='Reacher_Linux_NoVis/Reacher.x86_64')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations[0]
state_size = states.shape[0]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

agent = Agent(state_size,action_size,10)

def ddpg(n_episodes=500):
    scores_deque = deque(maxlen=n_episodes)
    scores = []
    max_score = -np.Inf
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        
        agent.reset()
        score = 0
        while True:
            actions = agent.act(state,add_noise=False)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_state = env_info.vector_observations[0]        # get next state (for each agent)
            rewards = env_info.rewards[0]                       # get reward (for each agent)
            done = env_info.local_done[0]                        # see if episode finished
            score += rewards                     # update the score (for each agent)         
            
            agent.step(state, actions, rewards, next_state, done)
            state = next_state
            if done:
                break 
            
            
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tScore: {:.2f}'.format(i_episode, score))
                
        if (i_episode % 5) == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        
        if len(scores)>100:
            if np.mean(scores[-100:])>30:
                print("Solve the environnement")
                torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
                torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
                break
              
    return scores

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
