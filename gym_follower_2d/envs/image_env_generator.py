#!/usr/bin/python

import scipy
from scipy import stats
import numpy as np
from math import sqrt, asin, cos, sin, atan2,pi
import networkx as nx
from gym_follower_2d.envs.image_env_utils import *
from gym_follower_2d.envs.geometry_utils import *
from gym_follower_2d.envs.env_generator import *
import sys
import pickle
import cv2
from os import path
import math
import sys

class ImageEnvironmentGenerator(EnvironmentGenerator):

    def __init__(self, x_range, y_range):
        self.x_range = x_range
        self.y_range = y_range

    def get_paths(self,paths_path):
        data = pickle.load(open(paths_path,'rb'),encoding='bytes')
        paths = ([(path[2]) for path in data[b'paths'] ])
        w = self.x_range[1] - self.x_range[0]
        h = self.y_range[1] - self.y_range[0]
        return np.array([np.array([np.multiply(np.array(i),np.array([w,-h,1]) ) + np.array([0,h,pi/2.0]) for i in path ]) for path in paths])

    def generate_from_image(self,feature_path):
        
        def populate_grid(img,grid):
            for i,x in enumerate(img):
                for j,y in enumerate(x):
                    if y <= 230:
                        grid[i][j] = 0
        
        def get_outlines(grid,x_ratio,y_ratio):
            paded_shape = (grid.shape[0]+2,grid.shape[1]+2)
            temp_grid = np.ones(paded_shape) * 255
            border_grid = np.ones(paded_shape) * 255
            temp_grid[1:paded_shape[0]-1,1:paded_shape[1]-1] = grid
            visited = {}

            def get_neighbours(cell):
                neighbours = []
                for i in [-1,0,1]:
                    for j in [-1,0,1]:
                        if i != 0 or j != 0:
                            temp = (cell[0] + i, cell[1] + j)
                            neighbours.append(temp)
                return neighbours

            def helper(start,id):
                queue = []
                queue.append(start)
                while queue != []:
                    curr = queue.pop(0)
                    visited[curr] = True
                    for i in get_neighbours(curr):
                        if temp_grid[i] == 255:
                            border_grid[curr] = id
                        if not i in visited:
                            if temp_grid[i] != 255:
                                visited[i] = True
                                queue.append(i) 
                pass

            def get_center(obs):
                avg_x = 0
                avg_y = 0
                for i in obs:
                    avg_x += i[1]
                    avg_y += i[0]
                return(int(avg_y/len(obs)), int(avg_x/len(obs)))

            def get_obstacles(max_id):
                obstacles = [[] for _ in range(max_id)]
                for i,x in enumerate(border_grid):
                    for j,y in enumerate(x):
                        temp = (i,j)
                        if border_grid[temp] != 255:
                            obstacles[int(border_grid[temp])].append((int(j*x_ratio),-int(i*y_ratio) + self.y_range[1]))
                for i,x in enumerate(obstacles):
                    origin = get_center(x)
                    refvec = [0, 1]
                    def clockwiseangle_and_distance(point):
                        vector = [point[0]-origin[0], point[1]-origin[1]]
                        lenvector = math.hypot(vector[0], vector[1])
                        if lenvector == 0:
                            return -math.pi, 0
                        normalized = [vector[0]/lenvector, vector[1]/lenvector]
                        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     
                        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1] 
                        angle = math.atan2(diffprod, dotprod)
                        if angle < 0:
                            return 2*math.pi+angle, lenvector
                        return angle, lenvector
                    obstacles[i] = ImageObstacle(sorted(x,key=clockwiseangle_and_distance))
                return obstacles

            count = 0
            for i,x in enumerate(temp_grid):
                for j,y in enumerate(x):
                    if (i,j) not in visited:
                        if y == 0:
                            helper((i,j),count)
                            count += 1
                        else:
                            visited[(i,j)] = True
            return get_obstacles(count)


        buildings = cv2.imread(path.join(feature_path,'mini_buildings_dilated_1.jpg'),0)
        roads = cv2.imread(path.join(feature_path,'mini_roads_dilated_1.jpg'),0)
        vegetation = cv2.imread(path.join(feature_path,'mini_vegetation_dilated_1.jpg'),0)
        trees = cv2.imread(path.join(feature_path,'mini_trees_dilated_1.jpg'),0)

        grid = np.ones(buildings.shape)*255

        x_ratio = float(self.x_range[1]/buildings.shape[1])
        y_ratio = float(self.y_range[1]/buildings.shape[0])

        populate_grid(buildings,grid)
        populate_grid(trees,grid)
        return get_outlines(grid,x_ratio,y_ratio),grid


class ImageEnvironmentCollection(EnvironmentCollection):

    def save(self, pkl_filename):
        file_object = open(pkl_filename, 'wb')
        worlds_without_classes = { idx : (world.x_range,
                                          world.y_range,
                                        world.obstacles,
                                        world.grid,
                                        world.paths)
                                    
                                    for idx, world in self.map_collection.items()}


        pickle.dump((self.x_range, self.y_range, worlds_without_classes), file_object)
        file_object.close()
        print()
        for idx in self.map_collection:
            world = self.map_collection[idx]
            cv2.imwrite(path.join(path.dirname(pkl_filename),'grid_{}.jpg'.format(idx)),world.image)

    def read(self, pkl_filename):
        file_object = open(pkl_filename, 'rb')
        self.x_range, self.y_range, worlds_without_classes = pickle.load(file_object, encoding='bytes')    
        self.map_collection = {idx: ImageEnvironment(val[0], val[1], val[2],val[3],val[4]) for idx, val in worlds_without_classes.items()}
        file_object.close()

    def generate_random(self, x_range, y_range, num_environments):
        self.x_range = x_range
        self.y_range = y_range
        self.num_environments = num_environments
        self.map_collection = {}

        eg = ImageEnvironmentGenerator(x_range, y_range)
        for i in range(self.num_environments):
            print('Sampling environment', i)
            obstacles,grid = eg.generate_from_image('C:\\Users\\gabri\\Documents\\McGill\\Robo Research\\fast_sampling_irl\data\\aerial_images\\map_' + str(i+1) +'\\feature_maps')
            trajectory_path = eg.get_paths('C:\\Users\\gabri\\Documents\\McGill\\Robo Research\\fast_sampling_irl\data\\aerial_images\\map_' + str(i+1) +'\\trajectories\\trajectories.pkl')
            self.map_collection[i] = ImageEnvironment(self.x_range, self.y_range, obstacles,grid,trajectory_path)

