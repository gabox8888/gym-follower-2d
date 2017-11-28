import scipy
from scipy import stats
import numpy as np
from math import sqrt, asin, cos, sin, atan2, ceil,pi
import networkx as nx
from gym_follower_2d.envs.geometry_utils import *
from gym_follower_2d.envs.env_utils import *
from random import randint
import sys
import pickle
import cv2


class ImageObstacle(object):
    def __init__(self, points):
        self.perimeterPoints = points

    def __eq__(self, other):
        return self.perimeterPoints == other.perimeterPoints

    def distance_to_point(self, x, y):
        p = np.array([x,y])
        dist = [point_to_point(p,np.array(i)) for i in self.perimeterPoints]
        return min(dist)

    def closest_point_to(self, p):
        closest_points_to_segments = [closest_point_on_segment(p, s, t) for c,w,h in zip(self.rectangle_centers, self.rectangle_widths, self.rectangle_heights) \
                                      for s,t in rectangle_edges( np.array([c[0] + w/2.0, c[1] + h/2.0]), \
                                                                  np.array([c[0] + w/2.0, c[1] - h/2.0]), \
                                                                  np.array([c[0] - w/2.0, c[1] - h/2.0]), \
                                                                  np.array([c[0] - w/2.0, c[1] + h/2.0]) )]

        distances = [np.linalg.norm(p - cp) for cp in closest_points_to_segments]
        idx = np.argmin(distances)

        return closest_points_to_segments[idx]

class ImageEnvironment(Environment):
    def __init__(self, x_range, y_range, obstacles,grid,paths):
        self.obstacles = obstacles
        
        self.x_range = x_range
        self.y_range = y_range
        self.grid = grid

        w = x_range[1] - x_range[0]
        h = y_range[1] - y_range[0]
        self.paths = paths
        self.path = self.paths[randint(0,len(paths)-1)]
        
        self.compute_occupancy_grid(w, h,grid)

    def __eq__(self, other):
        return self.obstacles == other.obstacles and self.x_range == other.x_range and self.y_range == other.y_range

    def reset_path(self,test=False):
        random_int = randint(0,int(0.8*(len(self.paths)-1))) if not test else randint(int(0.8*(len(self.paths)-1)),len(self.paths)-1)
        self.path = self.paths[random_int]

    def point_is_in_free_space(self, x, y, epsilon=0.25):
        row = int(x)
        col = int(y)

        if (row >=0 and row < self.image.shape[0] and col >= 0 and col < self.image.shape[1]):
            return (self.image[row, col, :] == (255, 255, 255)).all()
        else:
            return True

    def segment_is_in_free_space(self, x1,y1, x2,y2, epsilon=0.5):
        # Note: this is assuming that 1px = 1m
        a = np.array([x1,y1])
        b = np.array([x2,y2])
        L = np.linalg.norm(b-a)
        fs = [self.point_is_in_free_space(a[0] + i/L*(b[0]-a[0]), a[1] + i/L*(b[1]-a[1])) for i in range(ceil(L))]
        return all(fs)

    def compute_occupancy_grid(self, w, h,grid):
        self.image = 255*np.ones((h, w, 3), dtype='uint8')
        x_ratio = float(w/grid.shape[1])
        y_ratio = float(h/grid.shape[0])
        for i,x in enumerate(grid):
            for j,y in enumerate(x):
                if y == 0:
                    self.image[int(y_ratio*i):int(y_ratio*(i+1)), int(x_ratio*j):int(x_ratio*(j+1)), :] = (204, 153, 102)

    def segment_distance_from_obstacles(self, x1, y1, x2, y2):

        if not self.segment_is_in_free_space(x1, y1, x2, y2, epsilon=1e-10):
            return 0.0

        a = np.array([x1, y1])
        b = np.array([x2, y2])

        dist = [point_to_segment_distance(p, a, b) for p in obs.perimeterPoints for obs in self.obstacles]

        return min(dist)
                
    def point_distance_from_obstacles(self, x, y):
        dist = [obs.distance_to_point(x, y) for obs in self.obstacles]
        return min(dist)

    def rotate_pnt(self,pnts,theta,center):
        new_pnts = []
        theta *= -1
        theta += pi/2
        for i in pnts:
            translated = (i[0]-center[0],i[1]- center[1])
            rotated = (translated[0]*cos(theta) - translated[1]*sin(theta), translated[1]*cos(theta) + translated[0]*sin(theta))
            new_pnts.append([rotated[0] + center[0],rotated[1] + center[1]])
        return np.array(new_pnts)

    def get_img_observation(self,follower,target,fovr):
        height = 10
        width = 7
        converted_target = (target[0],self.image.shape[0] - target[1],target[2])
        converted_follower = (follower[0],self.image.shape[0] - follower[1],follower[2])
        tempImg = 255*np.ones((self.image.shape[0], self.image.shape[1], 3), dtype='uint8')
        target_triangle = np.array([[converted_target[0], converted_target[1]  - height],
                                    [converted_target[0] - width,converted_target[1] + height],
                                    [converted_target[0] + width,converted_target[1] + height]])
        follower_triangle = np.array([[converted_follower[0], converted_follower[1]  - height],
                                    [converted_follower[0] - width,converted_follower[1] + height],
                                    [converted_follower[0] + width,converted_follower[1] + height]])
        cv2.circle(tempImg,(int(converted_follower[0]),int(converted_follower[1])),200,(0,255,0),-1)
        cv2.fillConvexPoly(tempImg,np.int32([self.rotate_pnt(follower_triangle,converted_follower[2],converted_follower)]),(255,0,0))
        cv2.fillConvexPoly(tempImg,np.int32([self.rotate_pnt(target_triangle,converted_target[2],converted_target)]),(0,0,255))
        
        tempImg =  cv2.addWeighted(self.image,0.5,tempImg,0.5,0)
        tempImg = cv2.resize(tempImg, (0,0), fx=0.0625, fy=0.0625) 
        return tempImg


    def raytrace(self, p, theta,max_range,fov=2*pi,n_beams=16):
        p = (p[0],self.y_range[1] - p[1])
        theta *= -1

        degree_per_beam = float(fov/n_beams)
        curr_angle = theta - (fov/2.0)

        lidar_measurments = [1.0 for i in range(n_beams+1)]

        for beam in range(n_beams+1):
            for length in range(max_range):
                new_p = (int(length * sin(curr_angle) + p[1]),int(length * cos(curr_angle) + p[0]))
                if new_p[0] >= self.y_range[1] or new_p[1] >= self.x_range[1] or new_p[0] < 0 or new_p[1] < 0:
                    lidar_measurments[beam] = max(float((length - 1)/max_range),0.0)
                    break
                elif self.image[new_p][0] == 204 and self.image[new_p][1] ==  153 and self.image[new_p][2] == 102:
                    lidar_measurments[beam] = max(float(length/max_range),0.0)
                    break
                # else:
                #     self.image[new_p] = (234,54,20)
            curr_angle += degree_per_beam
    
        return lidar_measurments

        


    def winding_angle(self, path, point):
        wa = 0
        for i in range(len(path)-1):
            p = np.array([path[i].x, path[i].y])
            pn = np.array([path[i+1].x, path[i+1].y])

            vp = p - point
            vpn = pn - point

            vp_norm = sqrt(vp[0]**2 + vp[1]**2)
            vpn_norm = sqrt(vpn[0]**2 + vpn[1]**2)

            assert (vp_norm > 0)
            assert (vpn_norm > 0)

            z = np.cross(vp, vpn)/(vp_norm * vpn_norm)
            z = min(max(z, -1.0), 1.0)
            wa += asin(z)

        return wa

    def homology_vector(self, path):
        L = len(self.obstacles)
        h = np.zeros((L, 1) )
        for i in range(L):
            h[i, 0] = self.winding_angle(path, self.obstacles[i].representative_point)

        return h.reshape((L,))
