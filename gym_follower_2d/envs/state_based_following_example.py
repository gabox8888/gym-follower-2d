import gym
import gym_follower_2d

from env_generator import Environment, EnvironmentCollection
import numpy as np
import time
import timeit

env = gym.make('State-Based-Follower-2d-Map0-v0')

observation = env.reset()
dt1 = []
dt2 = []
total_reward = 0

for t in range(300):
    env.render()

    start = time.time()
    action = env.action_space.sample()
    action = np.array([0.0, 0.0])
    end = time.time()

    dt1.append(end - start)

    start = time.time()
    observation, reward, done, info = env.step(action)
    end = time.time()

    total_reward += reward
    
    dt2.append(end - start)

    
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break

print( sum(dt1)/len(dt1), sum(dt2)/len(dt2))
print ("Total reward", total_reward)

