import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_follower_2d.envs.env_utils import Environment 
from gym_follower_2d.envs.image_env_generator import ImageEnvironmentCollection 
from gym_follower_2d.envs.rrt import RRT,State
from gym.envs.classic_control import rendering
from gym.spaces import Box, Tuple

from math import pi, cos, sin, pi, atan2
import numpy as np
import random
import cv2
from types import MethodType

import os
import pyglet

class ImmitationBasedFollowing2DEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 worlds_pickle_filename=os.path.join(os.path.dirname(__file__), "assets", "worlds_640x480_v3.pkl"),
                 world_idx=1,
                 initial_follower_position = np.array([598.0, 297.0, 3.14]),
                 initial_target_position = np.array([588,297, 0.0]),
                 destinations = [],
                 target_paths = [],
                 max_observation_range = 100.0,
                 max_follower_speed = -2.7,
                 max_follower_theta = -0.15,
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

        self.rrt = RRT(self.world)
        
        self.max_observation_range = max_observation_range
        self.destination_tolerance_range = destination_tolerance_range
        self.viewer = None
        self.num_beams = 16
        self.max_follower_speed = max_follower_speed
        self.max_follower_theta = max_follower_theta
        self.iter_num = 0
        self.target_speed = 2.0
        
        self.add_self_position_to_observation = add_self_position_to_observation
        self.add_goal_position_to_observation = add_goal_position_to_observation

        assert not (self.destinations is None)
        self.init_follower_position = initial_follower_position
        self.init_target_position = initial_target_position
        self.state = np.concatenate((self.init_follower_position, self.init_target_position))
        # self.observation = self._get_observation(self.state)

        self.image_path = os.path.join(os.path.dirname(worlds_pickle_filename),'grid_{}.jpg'.format(world_idx))
        
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

    def _append_elements_to_viewer(self, viewer,
                                   screen_width,
                                   screen_height,
                                   obstacles,
                                   destination=None,
                                   destination_tolerance_range=None):

        viewer.set_bounds(left=0, right=screen_width, bottom=0, top=screen_height)

        image = rendering.Image(self.image_path,screen_width,screen_height)

        def renderSpoof1(self):
            sprite = pyglet.sprite.Sprite(self.img)
            sprite.draw()

        image.render1 = MethodType(renderSpoof1, image)
            
        viewer.add_geom(image)

        if not (destination is None):
            tr = rendering.Transform(translation=(destination[0], destination[1]))
            polygon = rendering.make_circle(radius=destination_tolerance_range, res=30, filled=True)
            polygon.add_attr(tr)
            polygon.set_color(1.0, 0., 0.)
            viewer.add_geom(polygon)
    
    def _reset(self):
        self.iter_num = 0
        self.world.reset_path(test=False)
        self.state[0:3] = self.world.path[self.iter_num]
        self.state[3:6] = self.world.path[self.iter_num]
        return self._get_observation(self.state)

    def _plot_state(self, viewer, state):
        follower_state = state[:3]
        target_state = state[3:]

        triangle_width = 4
        triangle_height = 7
        triangle = [(-triangle_height,triangle_width),(-triangle_height,-triangle_width),(triangle_height,0)]

        polygon = rendering.FilledPolygon(triangle)
        follower_tr = rendering.Transform(translation=tuple(follower_state[:2]), rotation=follower_state[2])
        polygon.add_attr(follower_tr)
        polygon.set_color(0.0,0.0,1.0)
        helperLine = rendering.Line(start=(0,0), end=(100,0))
        helperLine.add_attr(follower_tr)
        helperLine.set_color(0.0,0.0,1.0)
        viewer.add_onetime(helperLine)
        viewer.add_onetime(polygon)

        polygon = rendering.FilledPolygon(triangle)
        target_tr = rendering.Transform(translation=tuple(target_state[:2]), rotation=target_state[2])
        polygon.add_attr(target_tr)
        polygon.set_color(1.0,0.0,0.0)
        viewer.add_onetime(polygon)

    def _step(self,action):
        follower_state = self.state.copy()[:3]

        if action[2]:
            v = self.max_follower_speed*action[0]
            theta = self.max_follower_theta*action[1]
            
        else:
            iter_num_future = self.iter_num + 10
            start_state = State(self.state[0],self.state[1],None)
            end_state = self.world.path[int(iter_num_future /self.target_speed)] if int(iter_num_future/self.target_speed) < len(self.world.path) else self.world.path[-1]
            end_state = State(end_state[0],end_state[1],None)
            rrt_plan,_ = self.rrt.plan(start_state,end_state,100,10,10)
            if len(rrt_plan) > 1:
                delta = np.array([rrt_plan[1].x,rrt_plan[1].y]) - follower_state[:2]
                theta = atan2(delta[1],delta[0])
                v = max(-delta[0]/cos(theta),-self.max_follower_speed)
                theta = theta - follower_state[2]
                
            else:
                v = theta = 0
        
        dx = v*cos(follower_state[2])
        dy = v*sin(follower_state[2])
        follower_state += np.array([dx, dy,theta])

        mapped_x = max(min(int(follower_state[0]),self.world.x_range[1]-1),0)
        mapped_y = max(min(-(int(follower_state[1]) - self.world.y_range[1]),self.world.y_range[1]-1),0)
    
        if (self.world.image[mapped_y,mapped_x,:] == 255).all():
            self.state[:3] += np.array([dx, dy,theta])
            self.state[2] = self.state[2] % (2*pi)


        
        self.state[3:] =  self.world.path[int(self.iter_num/self.target_speed)] if int(self.iter_num/self.target_speed) < len(self.world.path) else self.world.path[-1]
        self.iter_num += 1

        return self._get_observation(self.state),None,False,None

    def _get_observation(self,state):
        prev_timestep = min(max(0,int((self.iter_num - self.target_speed)/self.target_speed)),len(self.world.path)-1)
        old_state = self.world.path[prev_timestep]
        follower_state = np.divide(state[:3],np.array([self.world.x_range[1],self.world.y_range[1],1]))
        target_state = np.divide(state[3:],np.array([self.world.x_range[1],self.world.y_range[1],1]))
        old_target = np.divide(old_state,np.array([self.world.x_range[1],self.world.y_range[1],1]))
        temp_target_state = follower_state - target_state
        old_diff = target_state - old_target
        target_v = old_diff[0]/(cos(old_target[2])*self.target_speed)
        target_w = old_diff[2]/self.target_speed
        return self.world.get_img_observation(state[:3],state[3:],200)
        # return np.array([sin(follower_state[2]), cos(follower_state[2]), target_v,target_w] + list(temp_target_state[:2]) + self.world.raytrace(self.state[:2],self.state[2],200,n_beams=32))
    
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
                                            obstacles=self.world.image,
                                            destination=None,
                                            destination_tolerance_range=self.destination_tolerance_range)

        self._plot_state(self.viewer, self.state)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

