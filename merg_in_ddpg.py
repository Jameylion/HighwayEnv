import time
import cv2
import gymnasium as gym
import numpy as np

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

import highway_env
from highway_env.vehicle.kinematics import Performance, Logger


if __name__ == "__main__":
    train = False
    situation = 'merge_in-v0'
    modelname = "DDPG"
    frameSize = (1280,560)
    out = cv2.VideoWriter('merge_in/videos/video_'+situation+"_"+modelname+'.avi', cv2.VideoWriter_fourcc(*'mp4v'), 16, frameSize)
    env = gym.make(situation, render_mode="rgb_array")

    env.configure({
    "screen_width": 1280,
    "screen_height": 560,
    "renderfps": 16
    })  # Higher FPS for rendering
    env.configure({
     "action": {
            "type": "ContinuousAction",
            "steering_range": [-np.pi / 3, np.pi / 3],
            "longitudinal": False,
            "lateral": True,
            "dynamical": True
            },
    
    })

    # The noise objects for DDPG
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=10)
    model.save("merge_in/model_"+modelname)
    vec_env = model.get_env()

    del model # remove to demonstrate saving and loading

    # Run the trained model and record video
    env = gym.make(situation, render_mode="rgb_array")
    model = DDPG.load("merge_in/model_"+modelname, env=env)



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
            # cur_frame = env.render()
            # out.write(cur_frame)

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

