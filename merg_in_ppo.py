import time
import cv2
from sb3_contrib import TRPO
import highway_env

import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env
from highway_env.vehicle.kinematics import Performance, Logger


if __name__ == "__main__":
    train = False
    situation = 'merge_in-v0'
    modelname = "PPO"
    frameSize = (1280,560)
    out = cv2.VideoWriter('merge_in/videos/video_'+situation+"_"+modelname+'.avi', cv2.VideoWriter_fourcc(*'mp4v'), 16, frameSize)
    if train:
        n_cpu = 6
        batch_size = 64
        env = make_vec_env(situation, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        model = PPO("MlpPolicy",
                    env,
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                    n_steps=batch_size * 12 // n_cpu,
                    batch_size=batch_size,
                    n_epochs=10,
                    learning_rate=5e-4,
                    gamma=0.8,
                    verbose=2,
                    tensorboard_log="merge_in_ppo/")
        
        model.learn(total_timesteps=int(10e4), progress_bar=True)
        model.save("merge_in/model_"+modelname)
        del model

    # Run the trained model and record video
    env = gym.make(situation, render_mode="rgb_array")
    model = TRPO.load("merge_in/model_"+modelname, env=env)

    env.configure({
    "screen_width": 1280,
    "screen_height": 560,
    "renderfps": 16
    })# Higher FPS for rendering
    env.configure({
    "simulation_frequency": 15,
    "policy_frequency":15
    }) 


    perfm = Performance()
    lolly = Logger()

    number_of_runs = 100
    for f in range(number_of_runs):
        done = truncated = False
        obs, info = env.reset()
        reward = 0

        ego_car = env.controlled_vehicles[0]

        stepcounter = 0
        
        while (not done) and ego_car.speed > 2 and stepcounter < 800:        
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            stepcounter += 1
            lolly.file(ego_car)
            cur_frame = env.render()
            out.write(cur_frame)

        perfm.add_measurement(lolly)
        lolly.clear_log()
        print(f)

    perfm.print_performance()
    print('DONE')

    number_of_collisions = 0
    T = 1
    while T < 10:
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)  # env.step(action.item(0))
            # print(action)
            # print(obs)
            # print(info)
            # print(reward)
            if info.get('crashed'):
                number_of_collisions += 1
            env.render()
            cur_frame = env.render()
            out.write(cur_frame)
        #print('crashrate is '+str(float(number_of_collisions)/T)+' and T is'+str(T))
        time.sleep(2)
        T+=1
    out.release()
    # print('number_of_collisions is:', number_of_collisions)
    print('DONE')

