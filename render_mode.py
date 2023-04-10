import gym
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', render_mode = 'ansi')
env.reset()
print(env.render())

env = gym.make('FrozenLake-v1', render_mode = 'rgb_array')
env.reset()
plt.imshow(env.render())

env = gym.make('FrozenLake-v1', render_mode = 'human')
env.reset()
env.render()

