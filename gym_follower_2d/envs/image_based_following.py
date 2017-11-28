import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_follower_2d.envs.env_generator import Environment, EnvironmentCollection
from gym.envs.classic_control import rendering
from gym.spaces import Box, Tuple

from gym_follower_2d.envs.range_based_following import StateBasedFollowing2DEnv
from math import pi, cos, sin, floor
import numpy as np
import cv2


class ImageBasedFollowing2DEnv(StateBasedFollowing2DEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, *args, **kwargs):
        StateBasedFollowing2DEnv.__init__(self, *args, **kwargs)
        self.obs_img_shape = (80, 60, 3)
        self.observation_space = Box(0., 255., (self.obs_img_shape[1], self.obs_img_shape[0], 3))

    def _get_observation(self, state, target_is_visible):
        image = self.world.image.copy()

        
        N = int(1*self.max_observation_range)
        delta_angle = 2*pi/N
        ranges = [self.world.raytrace(state[0:2],
                                      i * delta_angle,
                                      self.max_observation_range) for i in range(N)]

        
        delta_angle = 2*pi/N
        for i in range(len(ranges)):
            r = ranges[i]
            if r < 0:
                r = self.max_observation_range

            theta = i*delta_angle
            start = (int(state[0]), int(state[1]))
            end = (int(state[0] + r*cos(theta)), int(state[1] + r*sin(theta)))
            cv2.line(image, start, end, color=(255,255,0), thickness=1)
            
        follower_state_col = int(self.state[0])
        follower_state_row = int(self.state[1])

        dest_col = int(self.destination[0])
        dest_row = int(self.destination[1])

        cv2.circle(image, center=(follower_state_col, follower_state_row), radius=5, color=(0,0,0), thickness=-1)
        cv2.circle(image, center=(dest_col, dest_row), radius=int(self.destination_tolerance_range), color=(255,0,0), thickness=-1)

        if target_is_visible:
            target_state_col = int(self.state[2])
            target_state_row = int(self.state[3])
            cv2.circle(image, center=(target_state_col, target_state_row), radius=5, color=(0,255,0), thickness=-1)
        
            
        image = cv2.flip(image, flipCode=0)
        image = cv2.resize(image, self.obs_img_shape[0:2])
        return image
