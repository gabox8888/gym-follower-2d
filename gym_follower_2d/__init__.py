from gym.envs.registration import register
import numpy as np

# (x,y)
idx_to_goal = [ (np.array([230.0, 430.0]), np.array([230.0, 130.0]), np.array([500.0, 430.0]) )] * 10
idx_to_goal[1] = (np.array([130.0, 370.0]), np.array([130.0, 110.0]), np.array([520.0, 250.0]) )
idx_to_goal[2] = (np.array([530.0, 110.0]), np.array([130.0, 310.0]), np.array([460.0, 330.0]) )
idx_to_goal[3] = (np.array([400.0, 50.0]), np.array([180.0, 320.0]), np.array([430.0, 310.0]) )
idx_to_goal[4] = (np.array([180.0, 380.0]), np.array([610.0, 120.0]), np.array([420.0, 330.0]) )
idx_to_goal[5] = (np.array([500.0, 90.0]), np.array([180.0, 390.0]), np.array([380.0, 320.0]) )
idx_to_goal[6] = (np.array([480.0, 150.0]), np.array([440.0, 380.0]), np.array([310.0, 220.0]) )
idx_to_goal[7] = (np.array([500.0, 380.0]), np.array([470.0, 280.0]), np.array([270.0, 280.0]) )
idx_to_goal[8] = (np.array([250.0, 440.0]), np.array([420.0, 200.0]), np.array([150.0, 180.0]) )
idx_to_goal[9] = (np.array([390.0, 110.0]), np.array([520.0, 350.0]), np.array([240.0, 310.0]) ) 

for i in range(10):

    register(
        id='Limited-Range-Based-Follower-2d-Map%d-v0' % i,
        entry_point='gym_follower_2d.envs:LimitedRangeBasedFollowing2DEnv',
        max_episode_steps=1000,
        kwargs=dict(world_idx=i, destinations = idx_to_goal[i])
    )
    
"""
    register(
        id='Image-Based-Follower-2d-Map%d-v0' % i,
        entry_point='gym_follower_2d.envs:ImageBasedFollowing2DEnv',
        max_episode_steps=1000,
        kwargs=dict(world_idx=i, destinations = idx_to_goal[i])
    )
"""
    
