import highway_env

import gymnasium as gym

from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN


# from gymnasium.envs.registration import register
TRAIN = False

if __name__ == '__main__':
    # Create the environment
    # gym.register('merge_in-v0', entry_point='highway_env.envs:MergeinEnv')
    # highway_env.register_highway_envs()
    # print(gym.envs.registry.keys())
    # env = gym.make('merge_in-v0', render_mode="rgb_array")
    env = gym.make('merge_in-v0', render_mode="rgb_array")
    obs, info = env.reset()

    # Create the model
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                tensorboard_log="merge_in_dqn/")

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(2e4), progress_bar=True)
        model.save("merge_in/model")
        del model

    # Run the trained model and record video
    model = DQN.load("merge_in/model", env=env)
    env = RecordVideo(env, video_folder="merge_in/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({
    "screen_width": 1280,
    "screen_height": 560,
    "renderfps": 16
    })  # Higher FPS for rendering

    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()
