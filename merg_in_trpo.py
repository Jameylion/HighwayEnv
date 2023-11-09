import time
import cv2
from sb3_contrib import TRPO
import highway_env

import gymnasium as gym

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from highway_env.vehicle.kinematics import Performance, Logger


# from gymnasium.envs.registration import register
TRAIN = False

if __name__ == '__main__':
    # Create the environment
    # gym.register('merge_in-v0', entry_point='highway_env.envs:MergeinEnv')
    # highway_env.register_highway_envs()
    # print(gym.envs.registry.keys())
    # env = gym.make('merge_in-v0', render_mode="rgb_array")
    situation = 'merge_in-v0'
    modelname = "TRPO"
    frameSize = (1280,560)
    env = gym.make(situation, render_mode="rgb_array")
    obs, info = env.reset()

    out = cv2.VideoWriter('merge_in/videos/video_'+situation+"_"+modelname+'.avi', cv2.VideoWriter_fourcc(*'mp4v'), 16, frameSize)


    # Create the model
    model = TRPO("MlpPolicy", env,
            learning_rate=0.0003,
            n_steps=1024,
            batch_size=128,
            gamma=0.99,
            cg_max_steps=15,
            cg_damping=0.1,
            line_search_shrinking_factor=0.8,
            line_search_max_iter=10,
            n_critic_updates=10,
            gae_lambda=0.95,
            use_sde=False,
            sde_sample_freq=-1,
            normalize_advantage=True,
            target_kl=0.015,
            sub_sampling_factor=1,
            policy_kwargs=None,
            verbose=1,
            tensorboard_log="merge_in_TRPO/",
            seed=None,
            device='auto',
            _init_setup_model=True)

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(10e4), progress_bar=True)
        model.save("merge_in/model_"+modelname)
        del model

    # Run the trained model and record video
    model = TRPO.load("merge_in/model_"+modelname, env=env)
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
        T+=1

    out.release()
    # print('number_of_collisions is:', number_of_collisions)
    print('DONE')