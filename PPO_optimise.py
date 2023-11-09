from multiprocessing import freeze_support
from hyperopt import fmin, tpe, hp
import time
import cv2
from sb3_contrib import TRPO
import highway_env

import gymnasium as gym
from stable_baselines3 import PPO

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

def objective(params):
    # Extract hyperparameters
    n_cpu = params['n_cpu'] 
    batch_size = params['batch_size']
    net_arch = params['net_arch']
    n_steps = params['n_steps']
    n_epochs = params['n_epochs']
    learning_rate = params['learning_rate']
    gamma = params['gamma']

    # Create environment and train the model
    env = make_vec_env(situation, n_envs=8, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=net_arch),
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                gamma=gamma,
                verbose=0)
    
    model.learn(total_timesteps=int(2e4), log_interval=1000, progress_bar=False)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # Minimize the negative mean reward (maximize the reward)
    return -mean_reward

if __name__ == "__main__":
    freeze_support()
    train = False
    situation = 'merge_in-v0'
    modelname = "PPO"

   

    # Define the hyperparameter search space
    space = {
        'n_cpu': hp.choice('n_cpu', [4, 6, 8]),
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        'net_arch': hp.choice('net_arch', [[256, 256], [128, 128, 128]]),
        'n_steps': hp.choice('n_steps', [128, 256, 512]),
        'n_epochs': hp.choice('n_epochs', [5, 10, 20]),
        'learning_rate': hp.loguniform('learning_rate', -5, -3),
        'gamma': hp.uniform('gamma', 0.5, 0.99)
    }

    
    # Perform random search optimization
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10)

    print("Best hyperparameters:", best)