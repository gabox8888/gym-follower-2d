import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .env_generator import Environment, EnvironmentCollection
from gym.envs.classic_control import rendering
from gym.spaces import Box, Tuple

from math import pi, cos, sin
import numpy as np
import random

from rrt import State

import os

class LimitedRangeBasedFollowing2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 worlds_pickle_filename=os.path.join(os.path.dirname(__file__), "assets", "worlds_640x480_v2.pkl"),
                 world_idx=0,
                 initial_follower_position = np.array([-20.0, -20.0]),
                 initial_target_position = np.array([-15.0, -15.0]),
                 destinations = [],
                 target_paths = [],
                 max_observation_range = 100.0,
                 max_follower_speed = 10.0,
                 max_target_speed = 5.0,
                 destination_tolerance_range=20.0,
                 add_self_position_to_observation=False,
                 add_goal_position_to_observation=False):

        worlds = EnvironmentCollection()
        worlds.read(worlds_pickle_filename)

        target_path = np.concatenate((np.linspace(initial_target_position[0], destinations[0][0], 100),
                                      np.linspace(initial_target_position[1], destinations[0][1], 100))).reshape((2,100))

        
        target_path = [target_path[:, i] for i in range(100)]
        
        target_paths = [[target_path]]
        
        self.world = worlds.map_collection[world_idx]
        self.destinations = destinations
        self.target_paths = target_paths
        self._choose_random_path_for_target()
        
        self.max_observation_range = max_observation_range
        self.destination_tolerance_range = destination_tolerance_range
        self.viewer = None
        self.num_beams = 16
        self.max_follower_speed = max_follower_speed
        self.max_target_speed = max_target_speed
        
        self.add_self_position_to_observation = add_self_position_to_observation
        self.add_goal_position_to_observation = add_goal_position_to_observation
        self.set_initial_positions(initial_follower_position, initial_target_position)
        
        low = np.array([0.0, 0.0])
        high = np.array([self.max_follower_speed, 2*pi])
        self.action_space = Box(low, high)
        
        low = [-1.0] * self.num_beams
        high = [self.max_observation_range] * self.num_beams

        if add_self_position_to_observation:
            low.extend([-10000., -10000.]) # x and y coords
            high.extend([10000., 10000.])
        if add_goal_position_to_observation:
            low.extend([-10000., -10000.]) # x and y coords
            high.extend([10000., 10000.])
        self.observation_space = Box(np.array(low), np.array(high))
        self.observation = []

    def _choose_random_path_for_target(self):
        random_destination_idx = 0
        self.destination = self.destinations[random_destination_idx]
        self.target_path = self.target_paths[random_destination_idx][0]
        self.target_path_length_crossed = 0
        self.target_path_current_waypoint_idx = 0
        
    def set_initial_positions(self, init_follower_position, init_target_position):
        assert not (self.destinations is None)
        self.init_follower_position = init_follower_position
        self.init_target_position = init_target_position
        self.state = np.concatenate((self.init_follower_position, self.init_target_position))
        self.observation = self._get_observation(self.state)
        
    def _get_observation(self, state):
        delta_angle = 2*pi/self.num_beams
        ranges = [self.world.raytrace(self.state[0:2],
                                      i * delta_angle,
                                      self.max_observation_range) for i in range(self.num_beams)]

        ranges = np.array(ranges)
        if self.add_self_position_to_observation:
            ranges = np.concatenate([ranges, self.state[0:2]])
        if self.add_goal_position_to_observation:
            ranges = np.concatenate([ranges, self.destination])
        return ranges

    def _target_step(self):
        if self.target_path_current_waypoint_idx >= len(self.target_path)-1:
            # target does not move
            return

        v = random.betavariate(alpha=5, beta=1) * self.max_target_speed
        
        for i in range(self.target_path_current_waypoint_idx + 1, len(self.target_path)):
            
            next_wpt = self.target_path[i]
            dl = np.linalg.norm(next_wpt - self.state[2:4])
            
            if v < dl:
                self.target_path_length_crossed += v
                self.state[2:4] += v/dl * (next_wpt - self.state[2:4])
                break
            
            else:
                self.target_path_length_crossed += (v-dl)
                self.state[2:4] = next_wpt
                self.target_path_current_waypoint_idx += 1
                v = v - dl

    def target_is_visible(self, state):
        dist = np.linalg.norm(state[0:2] - state[2:4])
        return dist < self.max_observation_range and self.world.segment_is_in_free_space(state[0], state[1], state[2], state[3], epsilon=0.25)

    
    def _step(self, action):
        old_state = self.state.copy()

        v = action[0]
        theta = action[1]
        dx = v*cos(theta)
        dy = v*sin(theta)

        self.state[0:2] += np.array([dx, dy])
        self._target_step()
        
        reward = -1 # minus 1 for every timestep you're not in the goal
        done = False
        info = {}
        
        follower_is_in_free_space = True

        if np.linalg.norm(self.destination - self.state[2:4]) < self.destination_tolerance_range:
            done = True
            
        if not self.world.point_is_in_free_space(self.state[0], self.state[1], epsilon=0.25):
            reward = -5 # for hitting an obstacle
            follower_is_in_free_space = False
            
        if not self.world.segment_is_in_free_space(old_state[0], old_state[1],
                                                   self.state[0], self.state[1],
                                                   epsilon=0.25):
            reward = -5 # for hitting an obstacle
            follower_is_in_free_space = False
            

        if follower_is_in_free_space:
            if self.target_is_visible(self.state):
                reward = 1
            else:
                reward = -1

        self.observation = self._get_observation(self.state)
        return self.observation, reward, done, info


    def _reset(self):
        self.state[0:2] = self.init_follower_position
        self.state[2:4] = self.init_target_position
        self._choose_random_path_for_target()
        return self._get_observation(self.state)

    def _plot_state(self, viewer, state):
        polygon = rendering.make_circle(radius=5, res=30, filled=True)
        follower_tr = rendering.Transform(translation=(state[0], state[1]))
        polygon.add_attr(follower_tr)
        viewer.add_onetime(polygon)

        polygon = rendering.make_circle(radius=5, res=30, filled=True)
        target_tr = rendering.Transform(translation=(state[2], state[3]))
        polygon.add_attr(target_tr)
        viewer.add_onetime(polygon)
        

    def _plot_observation(self, viewer, state, observation):
        delta_angle = 2*pi/self.num_beams
        for i in range(len(observation)):
            r = observation[i]
            if r < 0:
                r = self.max_observation_range

            theta = i*delta_angle
            start = (state[0], state[1])
            end = (state[0] + r*cos(theta), state[1] + r*sin(theta))

            line = rendering.Line(start=start, end=end)
            line.set_color(.5, 0.5, 0.5)
            viewer.add_onetime(line)
        
    def _append_elements_to_viewer(self, viewer,
                                   screen_width,
                                   screen_height,
                                   obstacles,
                                   destination=None,
                                   destination_tolerance_range=None):

        viewer.set_bounds(left=-100, right=screen_width+100, bottom=-100, top=screen_height+100)

        L = len(obstacles)
        for i in range(L):

            obs = obstacles[i]
            for c,w,h in zip(obs.rectangle_centers, obs.rectangle_widths, obs.rectangle_heights):
                l = -w/2.0
                r = w/2.0
                t = h/2.0
                b = -h/2.0

                rectangle = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                tr = rendering.Transform(translation=(c[0], c[1]))
                rectangle.add_attr(tr)
                rectangle.set_color(.8,.6,.4)
                viewer.add_geom(rectangle)


        if not (destination is None):
            tr = rendering.Transform(translation=(destination[0], destination[1]))
            polygon = rendering.make_circle(radius=destination_tolerance_range, res=30, filled=True)
            polygon.add_attr(tr)
            polygon.set_color(1.0, 0., 0.)
            viewer.add_geom(polygon)

    def _render(self, mode='human', close=False):

        if close:
            if self.viewer is not None:
                self.viewer.close()
            self.viewer = None
            return

        screen_width = (self.world.x_range[1] - self.world.x_range[0])
        screen_height = (self.world.y_range[1] - self.world.y_range[0])

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self._append_elements_to_viewer(self.viewer,
                                            screen_width,
                                            screen_height,
                                            obstacles=self.world.obstacles,
                                            destination=self.destination,
                                            destination_tolerance_range=self.destination_tolerance_range)

        self._plot_state(self.viewer, self.state)
        self._plot_observation(self.viewer, self.state, self.observation)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

"""
class StateBasedMDPNavigation2DEnv(LimitedRangeBasedPOMDPNavigation2DEnv):
    def __init__(self, *args, **kwargs):
        LimitedRangeBasedPOMDPNavigation2DEnv.__init__(self, *args, **kwargs)
        low = [-float('inf'), -float('inf'), 0.0, 0.0]
        high = [float('inf'), float('inf'), float('inf'), 2*pi]

        if self.add_goal_position_to_observation:
            low.extend([-10000., -10000.]) # x and y coords
            high.extend([10000., 10000.])

        self.observation_space = Box(np.array(low), np.array(high))

    def _plot_observation(self, viewer, state, observation):
        pass

    def _get_observation(self, state):
        # return state
        dist_to_closest_obstacle, absolute_angle_to_closest_obstacle = self.world.range_and_bearing_to_closest_obstacle(state[0], state[1])
        obs = np.array([state[0], state[1], dist_to_closest_obstacle, absolute_angle_to_closest_obstacle])
        if self.add_goal_position_to_observation:
            obs = np.concatenate([obs, self.destination])
        return obs
"""