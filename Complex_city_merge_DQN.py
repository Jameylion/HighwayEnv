import cv2
import gymnasium as gym

from gymnasium.envs.registration import register

from stable_baselines3 import PPO,DQN
from sb3_contrib import TRPO
import torch
import tensorflow
from highway_env.vehicle.kinematics import Performance, Logger
import highway_env
highway_env.register_highway_envs()
register(
    id='complex-city-v0',
    entry_point='highway_env.envs:ComplexcityEnv',
    )   

#situation = "intersection-v1"
env_name = "complex-city-v0"
#situation = "racetrack-v0"
# situation = "merge_in-v0" 
modelname = "DQN"


frameSize = (1280,560)
# out = cv2.VideoWriter('video'+situation+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 16, frameSize)
out = cv2.VideoWriter('merge_in/videos/video_'+env_name+"_"+modelname+'.avi', cv2.VideoWriter_fourcc(*'mp4v'), 16, frameSize)


env = gym.make(env_name, render_mode="rgb_array")


# Run the trained model and record video
model = DQN.load("merge_in/model_"+modelname, env=env)
# env = RecordVideo(env, video_folder="merge_in/videos", episode_trigger=lambda e: True)
# env.unwrapped.set_record_video_wrapper(env)
env.configure({
"screen_width": 1280,
"screen_height": 560,
"renderfps": 16
})# Higher FPS for rendering
env.configure({
"simulation_frequency": 15,
"policy_frequency":15
}) 

# for videos in range(10):
#     done = truncated = False
#     obs, info = env.reset()
#     while not (done or truncated):
#         # Predict
#         action, _states = model.predict(obs, deterministic=True)
#         # Get reward
#         obs, reward, done, truncated, info = env.step(action)
#         # Render
#         env.render()
# env.close()


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
  
    T+=1

out.release()
# print('number_of_collisions is:', number_of_collisions)
print('DONE')
