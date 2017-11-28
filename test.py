#!/usr/bin/python

import scipy
from scipy import stats
import numpy as np
from math import sqrt, asin, cos, sin, atan2
import networkx as nx
from gym_follower_2d.envs.image_env_generator import *
from gym_follower_2d.envs.env_utils import *
from gym_follower_2d.envs.geometry_utils import *
import sys
import pickle

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python env_generator.py filename_to_save.pkl")
        sys.exit(0)

    x_range=[0, 640]
    y_range=[0, 480]
    num_environments = 4

    ec = ImageEnvironmentCollection()
    ec.generate_random(x_range, y_range, num_environments)
    ec.save(sys.argv[1])
