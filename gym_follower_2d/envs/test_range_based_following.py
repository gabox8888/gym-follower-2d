import unittest
import numpy as np

from env_utils import Obstacle
from math import sqrt, atan2
import cv2
import os

from range_based_following import LimitedRangeBasedFollowing2DEnv


class TestLimitedRangeFollowingEnv(unittest.TestCase):

    

    def test_rewards_empty_env(self):
        
        destinations = (np.array([230.0, 430.0]), np.array([230.0, 130.0]), np.array([500.0, 430.0]) )
        env = LimitedRangeBasedFollowing2DEnv(world_idx=0, destinations=destinations)
        env.world.obstacles = [] 

        env._sample_target_speed = lambda : env.max_target_speed 

        self.assertEqual(env._sample_target_speed(), env.max_target_speed)
        self.assertTrue(env.target_is_visible(env.state))

        dist_left = env.max_observation_range - np.linalg.norm(env.state[0:2]- env.state[2:4])
        its_left = int(dist_left/env.max_target_speed)
        
        for i in range(its_left):
            observation, reward, done, info = env._step(action=np.array([0.0, 0.0]))
            if (i + 2) * env.max_target_speed < env.max_observation_range:
                self.assertTrue(env.target_is_visible(env.state))
                self.assertEqual(reward, 1)
                
        observation, reward, done, info = env._step(action=np.array([0.0, 0.0]))
        observation, reward, done, info = env._step(action=np.array([0.0, 0.0]))
        
        self.assertFalse(env.target_is_visible(env.state))
        self.assertEqual(reward, -1)

        
        theta = atan2(env.state[3] - env.state[1], env.state[2] - env.state[0])
        observation, reward, done, info = env._step(action=np.array([dist_left, theta]))

        self.assertTrue(env.target_is_visible(env.state))
        self.assertEqual(reward, 1)

    
    def test_rewards_one_obs_env(self):
        destinations = (np.array([400.0, 12.0]), )

        env = LimitedRangeBasedFollowing2DEnv(world_idx=0,
                                              destinations=destinations,
                                              initial_follower_position=np.array([5.0, 0.0]),
                                              initial_target_position=np.array([5.0, 12.0]) )
        
        centers = [np.array([12.0, 6.0])]
        widths = [10]
        heights = [10]
        env.world.obstacles = [Obstacle(centers, widths, heights)] 
        env.world.compute_occupancy_grid(env.world.image.shape[1], env.world.image.shape[0])

        env._sample_target_speed = lambda : env.max_target_speed 

        self.assertEqual(env._sample_target_speed(), env.max_target_speed)
        self.assertTrue(env.target_is_visible(env.state))
        
        observation, reward, done, info = env._step(action=np.array([0.0, 0.0]))
        
        self.assertFalse(env.target_is_visible(env.state))
        self.assertEqual(reward, -1)


    def test_rewards_one_obs_env_low_speed(self):
        destinations = (np.array([400.0, 12.0]), )

        env = LimitedRangeBasedFollowing2DEnv(world_idx=0,
                                              destinations=destinations,
                                              initial_follower_position=np.array([5.0, 0.0]),
                                              initial_target_position=np.array([5.0, 12.0]) )
        
        centers = [np.array([12.0, 6.0])]
        widths = [10]
        heights = [10]
        env.world.obstacles = [Obstacle(centers, widths, heights)] 
        env.world.compute_occupancy_grid(env.world.image.shape[1], env.world.image.shape[0])

        env.max_target_speed = 1.0
        env._sample_target_speed = lambda : env.max_target_speed 

        self.assertEqual(env._sample_target_speed(), env.max_target_speed)
        self.assertTrue(env.target_is_visible(env.state))
        
        observation, reward, done, info = env._step(action=np.array([0.0, 0.0]))
        
        self.assertTrue(env.target_is_visible(env.state))
        self.assertEqual(reward, 1)


    def test_rewards_thin_obs(self):
        destinations = (np.array([400.0, 12.0]), )

        env = LimitedRangeBasedFollowing2DEnv(world_idx=0,
                                              destinations=destinations,
                                              initial_follower_position=np.array([5.0, 0.0]),
                                              initial_target_position=np.array([5.0, 12.0]) )
        
        centers = [np.array([10.0, 6.0])]
        widths = [3]
        heights = [10]
        env.world.obstacles = [Obstacle(centers, widths, heights)] 
        env.world.compute_occupancy_grid(env.world.image.shape[1], env.world.image.shape[0])

        env._sample_target_speed = lambda : env.max_target_speed 

        self.assertEqual(env._sample_target_speed(), env.max_target_speed)
        self.assertTrue(env.target_is_visible(env.state))
        
        observation, reward, done, info = env._step(action=np.array([0.0, 0.0]))
        
        self.assertFalse(env.target_is_visible(env.state))
        self.assertEqual(reward, -1)

        
        
        
if __name__ == '__main__':
    unittest.main()
