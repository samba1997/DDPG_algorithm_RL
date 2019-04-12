import gym
env = gym.make('Pendulum-v0')
action_space = env.reset()
print(action_space)