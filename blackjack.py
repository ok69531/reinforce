#%%
import gym

env = gym.make('Blackjack-v1')
state = env.reset()
action = 1
env.step(action)

# env.action_space


env = gym.make('Blackjack-v1', render_mode = 'human')
state = env.reset()[0]

episode = []

num_timesteps = 20

for i in range(num_timesteps):
    random_action = env.action_space.sample()
    new_state, reward, done, _, _ = env.step(random_action)
    episode.append((state, action, reward))
    
    if done:
        break
    
    state = new_state

print(episode)


#%%
import gym
import pandas as pd
import random
from collections import defaultdict

env = gym.make('Blackjack-v1', render_mode = 'human')
Q = defaultdict(float)
total_return = defaultdict(float)
N = defaultdict(int)

def epsilon_greedy(state, Q):
    epsilon = 0.2
    
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key = lambda x: Q[(state, x)])

num_timesteps = 50

def generate_epsiode(Q):
    episode = []
    state = env.reset()[0]
    
    for t in range(num_timesteps):
        action = epsilon_greedy(state, Q)
        next_state, reward, done, _, _ = env.step(action)
        episode.append((state, action, reward))
        
        if done:
            break
        
        state = next_state
        
    return episode

num_episodes = 50000
for i in range(num_episodes):
    episode = generate_epsiode(Q)
    all_state_action_pairs = [(s, a) for (s, a, r) in episode]
    rewards = [r for (s, a, r) in episode]
    
    for t, (state, action, _) in enumerate(episode):
        if not (state, action) in all_state_action_pairs[:t]:
            G = sum(rewards[t:])
            total_return[(state, action)] = total_return[(state, action)] + G
            N[(state, action)] += 1
            Q[(state, action)] = total_return[(state, action)]/N[(state, action)]


df = pd.DataFrame(Q.items(), columns = ['state_action_pair', 'Q_value'])
df.head(10)
