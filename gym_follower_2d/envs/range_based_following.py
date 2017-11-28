import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_follower_2d.envs.env_utils import Environment 
from gym_follower_2d.envs.env_generator import EnvironmentCollection
from gym_follower_2d.envs.image_env_generator import ImageEnvironmentCollection 
from gym.envs.classic_control import rendering
from gym.spaces import Box, Tuple

from math import pi, cos, sin
import numpy as np
import random


import os

class LimitedRangeBasedFollowing2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 worlds_pickle_filename=os.path.join(os.path.dirname(__file__), "assets", "worlds_640x480_v3.pkl"),
                 world_idx=0,
                 initial_follower_position = np.array([50.0, 10.0]),
                 initial_target_position = np.array([50.0, 30.0]),
                 destinations = [],
                 target_paths = [],
                 max_observation_range = 100.0,
                 max_follower_speed = 10.0,
                 max_target_speed = 5.0,
                 destination_tolerance_range=20.0,
                 add_self_position_to_observation=False,
                 add_goal_position_to_observation=False):
        worlds = ImageEnvironmentCollection()
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

        assert not (self.destinations is None)
        self.init_follower_position = initial_follower_position
        self.init_target_position = initial_target_position
        self.state = np.concatenate((self.init_follower_position, self.init_target_position))
        #self.observation = self._get_observation(self.state, target_is_visible=True)
        
        low = np.array([0.0, 0.0])
        high = np.array([self.max_follower_speed, 2*pi])
        self.action_space = Box(low, high)


        low = [-1.0] * (self.num_beams + 3)
        high = [self.max_observation_range] * (self.num_beams + 3)

        if add_self_position_to_observation:
            low.extend([-10000., -10000.]) # x and y coords
            high.extend([10000., 10000.])
            
        if add_goal_position_to_observation:
            low.extend([-10000., -10000.] * len(destinations) ) # x and y coords
            high.extend([10000., 10000.] * len(destinations) )
            
        self.observation_space = Box(np.array(low), np.array(high))
        self.observation = []

    def _choose_random_path_for_target(self):
        random_destination_idx = 0
        self.destination = self.destinations[random_destination_idx]
        self.target_path = self.target_paths[random_destination_idx][0]
        self.target_path_length_crossed = 0
        self.target_path_current_waypoint_idx = 0
        
    def _get_observation(self, state, target_is_visible):
        delta_angle = 2*pi/self.num_beams
        ranges = [self.world.raytrace(state[0:2],
                                      i * delta_angle,
                                      self.max_observation_range) for i in range(self.num_beams)]

        obs_vector = np.array(ranges)

        if target_is_visible:
            obs_vector = np.concatenate([ obs_vector, state[2:4], np.array([1.0]) ])
        else:
            obs_vector = np.concatenate([ obs_vector, np.array([-1,-1]), np.array([0.0]) ])
        
            
        if self.add_self_position_to_observation:
            obs_vector = np.concatenate([obs_vector, state[0:2]])
            
        if self.add_goal_position_to_observation:
            obs_vector = np.concatenate([obs_vector] + self.destinations)

        return obs_vector

    def _sample_target_speed(self):
        return random.betavariate(alpha=5, beta=1) * self.max_target_speed
    
    def _target_step(self):
        if self.target_path_current_waypoint_idx >= len(self.target_path)-1:
            # target does not move
            return

        v = self._sample_target_speed()
        
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
        dist_ok = dist < self.max_observation_range
        fs_ok = self.world.segment_is_in_free_space(state[0], state[1], state[2], state[3])
        return dist_ok and fs_ok 

    
    def _step(self, action):

        old_state = self.state.copy()

        v = action[0]
        theta = action[1]
        dx = v*cos(theta)
        dy = v*sin(theta)

        #print (v, theta)
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
            

        target_is_visible = False
        if follower_is_in_free_space:
            target_is_visible = self.target_is_visible(self.state)
            if target_is_visible:
                reward = 1
            else:
                reward = -1

        self.observation = self._get_observation(self.state, target_is_visible)
        return self.observation, reward, done, info


    def _reset(self):
        self.state[0:2] = self.init_follower_position
        self.state[2:4] = self.init_target_position
        self._choose_random_path_for_target()
        return self._get_observation(self.state, target_is_visible=True)

    def _plot_state(self, viewer, state):
        polygon = rendering.make_circle(radius=5, res=30, filled=True)
        follower_tr = rendering.Transform(translation=(state[0], state[1]))
        polygon.add_attr(follower_tr)
        viewer.add_onetime(polygon)

        polygon = rendering.make_circle(radius=5, res=30, filled=True)
        target_tr = rendering.Transform(translation=(state[2], state[3]))
        polygon.add_attr(target_tr)
        polygon.set_color(0.0,1.0,0.0)
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

        viewer.set_bounds(left=-10, right=screen_width+10, bottom=-10, top=screen_height+10)

        L = len(obstacles)
        for i in range(L):

            obs = obstacles[i].perimeterPoints
            # for c,w,h in zip(obs.rectangle_centers, obs.rectangle_widths, obs.rectangle_heights):
            #     l = -w/2.0
            #     r = w/2.0
            #     t = h/2.0
            #     b = -h/2.0

            shape = rendering.FilledPolygon(obs)

            #rectangle = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            # tr = rendering.Transform(translation=(obs[1], obs[0]))
            # rectangle.add_attr(tr)
            shape.set_color(.8,.6,.4)
            viewer.add_geom(shape)


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

class StateBasedFollowing2DEnv(LimitedRangeBasedFollowing2DEnv):
    def __init__(self, *args, **kwargs):
        LimitedRangeBasedFollowing2DEnv.__init__(self, *args, **kwargs)
        
        #self.rectangles = [(c[0], c[1], w,h) for obs in self.world.obstacles for c, w, h in zip(obs.rectangle_centers,
                                                                                                #obs.rectangle_widths,
                                                                                                #obs.rectangle_heights)]

        #self.rectangle_obs_vector = [el for rect in self.rectangles for el in rect]

        
        inf = 100000.0
        low = [-inf, -inf, 0.0, 0.0, -inf, -inf, 0.0] 
        high = [inf, inf, inf, 2*pi, inf, inf, 1.0]

        #low.extend([-inf, -inf, self.world.x_range[0], self.world.y_range[0]] * len(self.rectangles))
        #high.extend([inf,  inf, self.world.x_range[1], self.world.y_range[1]] * len(self.rectangles))

        
        if self.add_goal_position_to_observation:
            low.extend([-10000., -10000.] * len(self.destinations)) # x and y coords
            high.extend([10000., 10000.] * len(self.destinations))

            
        self.observation_space = Box(np.array(low), np.array(high))

        
    def _plot_observation(self, viewer, state, observation):
        pass

    
    def _get_observation(self, state, target_is_visible):
        dist_to_closest_obstacle, absolute_angle_to_closest_obstacle = self.world.range_and_bearing_to_closest_obstacle(state[0], state[1])

        if target_is_visible:
            obs =  [state[0], state[1], dist_to_closest_obstacle, absolute_angle_to_closest_obstacle, state[2], state[3], 1.0] 
        else:
            obs =  [state[0], state[1], dist_to_closest_obstacle, absolute_angle_to_closest_obstacle, -1.0, -1.0, 0.0] 


        obs.extend(self.rectangle_obs_vector)

        obs = np.array(obs)

        if self.add_goal_position_to_observation:
            obs = np.concatenate([obs] + self.destinations)

        return obs

