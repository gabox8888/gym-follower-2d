import os

import gym
import gym_follower_2d
from gym_follower_2d.envs.env_generator import Environment, EnvironmentCollection

import numpy as np
import time
import cv2
import pygame
import pickle

from math import pi
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Dropout,Flatten,MaxPooling2D
import keras

from sklearn import svm

offset = 0
test = False
curr_v = "m"
map_v = "0"
novelty = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
iter_as_human = 100

def build_model():
    
    model = Sequential()

    # model.add(Dense(units=60, activation='relu', input_dim=39))
    # model.add(Dense(units=60, activation='relu'))
    # model.add(Dense(units=2,activation='tanh'))

    # model.add(Dense(units=100, activation='relu', input_dim=39))
    # model.add(Dense(units=60, activation='relu'))
    # model.add(Dense(units=70, activation='relu'))
    # model.add(Dense(units=2,activation='tanh'))

    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(30,40,3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=2, activation='tanh'))

    model.compile(loss=keras.losses.mean_squared_error,optimizer='adam')

    return model

def train_model(model,obs,act,i,epochs=50):
    if not test:
        model.fit(obs, act, epochs=epochs, batch_size=70)
        model.save('.\\data\\model_{}_{}.h5'.format(curr_v,i + offset)) 
        return model
    else:
        return load_model('.\\data\\model_{}_{}.h5'.format(curr_v,i+ offset))

def save_data(data):
    if not test:
        file_obj = open(".\\data\\rollout_{}.pkl".format(curr_v), 'wb')
        pickle.dump(data, file_obj)
        file_obj.close()

pygame.display.init()
pygame.joystick.init()
pygame.joystick.Joystick(0).init()
model = build_model() if offset == 0 else load_model('.\\data\\model_{}_{}.h5'.format(curr_v,offset))

env = gym.make('Immitation-Based-Follower-2d-Map{}-v0'.format(map_v))

override = False

if not test and offset == 0:
    data = {}
elif not test and offset > 0:
    data = pickle.load(open(".\\data\\rollout_{}.pkl".format(curr_v), 'rb'))
else:
    data = {}

# print(sorted([i for i in data]))

for i in range(50):
    is_teleop = not test
    observation = env.reset()
    print("Test2")
    obseravtions = []
    actions = []
    curr_as_human = 0
    for t in range(2000):
        done = False
        print("Test3")

        for event in pygame.event.get(): # User did something
            print("Test5")
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 1:
                   is_teleop = not is_teleop
                if event.button == 2:
                    save_data(data)
                    exit()
                if event.button == 7:
                    done = True
                if event.button == 6:
                    override = True
            elif event.type == pygame.JOYBUTTONUP:
                if event.button == 6:
                    override = False
            else:
                None
        print("Test4")


        # if i != 0 and not test: 
        #     if override:
        #         is_teleop = True
        #     if novelty.predict([observation])[0] == 1 and is_teleop and curr_as_human >= iter_as_human:
        #         curr_as_human = 0
        #         print("ROBOT")
        #         is_teleop = False
        #     elif novelty.predict([observation])[0] == -1 and not is_teleop:
        #         print("HUMAN")
        #         is_teleop = True
        # elif not test:
        #     is_teleop = True
        
        if not is_teleop:
            pred = model.predict(np.array([observation]))[0]
            up_down = pred[0]
            turn = pred[1]
        else:
            curr_as_human += 1
            up_down = pygame.joystick.Joystick(0).get_axis(1)
            turn =  pygame.joystick.Joystick(0).get_axis(2)
            
        up_down = up_down if abs(up_down) > 0.01 else 0.0
        turn =  turn if abs(turn) > 0.01 else 0.0

        action = np.array([up_down,turn,True])

        env.render()

        if is_teleop:
            actions.append(action[:2])
            obseravtions.append(observation)

        observation, reward, _, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps, at iteration: {}".format(t+1,i))
            break

    if not test:
        data[i + offset] = (actions,obseravtions)
        temp_o = []
        temp_a = []
        for t in range(i+1 if offset ==0 else i + offset ):
            temp_a += data[t][0]
            temp_o += data[t][1]
        actions = np.array(temp_a)
        obseravtions = np.array(temp_o)
        model = train_model(model,obseravtions,actions,i +1)
        # novelty.fit(obseravtions)
        continue
    else:
        model = train_model(model,[],[],i+1)
    save_data(data)