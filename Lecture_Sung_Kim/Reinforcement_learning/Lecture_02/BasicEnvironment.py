# This Python file uses the following encoding: utf-8

# https://youtu.be/xvDAURQVDhk?list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG
# RL lab 02 - Playing OpenAI GYM Games


import gym

env = gym.make("FrozenLake-v0")

observation = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

