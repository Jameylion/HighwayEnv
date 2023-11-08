import time
import cv2
import gymnasium as gym
import highway_env
from stable_baselines3 import PPO, SAC
from sb3_contrib import TRPO
import torch
import tensorflow
from highway_env.vehicle.kinematics import Performance, Logger

import warnings

# Filter out the specific warning
warnings.filterwarnings("ignore", category=UserWarning)

#situation = "intersection-v1"
model_name = "merge_in-v1"
#situation = "racetrack-v0"
situation = "merge_in-v1" 


frameSize = (1280,560)
# out = cv2.VideoWriter('video'+situation+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 16, frameSize)
out = cv2.VideoWriter('video'+situation+'sac.avi', cv2.VideoWriter_fourcc(*'mp4v'), 16, frameSize)


env = gym.make(situation, render_mode="rgb_array")
env.configure({
    "screen_width": 1280,
    "screen_height": 560,
    "renderfps": 16
})
env.configure({
    "action": {
        "type": "ContinuousAction"
    },
})
env.configure({
    "simulation_frequency": 15,
    "policy_frequency":15
})

env.reset()
# model = SAC("MlpPolicy", env,
#             learning_rate=0.0003,
#             batch_size=128,
#             gamma=0.9,
#             tensorboard_log="highway_SAC/",
#             device='cpu',
#             _init_setup_model=True)


# # uncomment the lines below if you want to train a new model

# # model = TRPO.load(situation+'_trpo/fixed_test')


# # model.set_env(env)
# # model.set_parameters(params)#, exact_match=True)

# print('learning....')
# model.learn(int(10000),progress_bar=True)
# print('done!')
# name = '_sac'
# model.save(situation+name)

# print()
# print(situation+name+" is saved!!")
# print()



# ########## Load and test saved model##############
# # model = TRPO.load(model_name +'_trpo/baseline_discrete')
# #while True:

model = SAC.load('merge_in-v1_sac')

perfm = Performance()
lolly = Logger()

number_of_runs = 10
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
        env.render()
        

    perfm.add_measurement(lolly)
    lolly.clear_log()
    print(f)

perfm.print_performance()
print('DONE')

# number_of_collisions = 0
# T = 1
# while True:
#     done = truncated = False
#     obs, info = env.reset()
#     while not (done or truncated):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, truncated, info = env.step(action)  # env.step(action.item(0))
#         # print(action)
#         # print(obs)
#         # print(info)
#         # print(reward)
#         if info.get('crashed'):
#             number_of_collisions += 1
#         env.render()
#         cur_frame = env.render()
#         out.write(cur_frame)
#     #print('crashrate is '+str(float(number_of_collisions)/T)+' and T is'+str(T))
#     time.sleep(2)
#     T+=1

# out.release()
# # print('number_of_collisions is:', number_of_collisions)
# print('DONE')

number_of_collisions = 0
T = 1
while True:
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
