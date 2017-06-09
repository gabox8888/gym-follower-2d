import gym
import gym_follower_2d

from env_generator import Environment, EnvironmentCollection
import numpy as np
import time
import cv2

env = gym.make('Image-Based-Follower-2d-Map0-v0')

observation = env.reset()
for t in range(1000):
    env.render()

    #action = env.action_space.sample()
    action = np.array([.0,.0])
    observation, reward, done, info = env.step(action)

    obs_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', obs_bgr)
    cv2.waitKey(10)

    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
