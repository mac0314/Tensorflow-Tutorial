# This Python file uses the following encoding: utf-8

# https://youtu.be/MF_Wllw9VKk?list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG
# RL lab 06 - Q Network for Cart Pole

import gym

'''
# Basic format
env = gym.make('CartPole-v0')
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
'''


env = gym.make('CartPole-v0')
env.reset()

random_episodes = 0
reward_sum = 0

while random_episodes < 10:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)

    print (observation, reward, done)

    reward_sum += reward
    if done:
        random_episodes += 1

        print ("Reward for this episode was: ", reward_sum)
        reward_sum = 0
        env.reset()


