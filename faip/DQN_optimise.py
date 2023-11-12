from multiprocessing import freeze_support
from hyperopt import fmin, tpe, hp
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

def objective(params):
    # Extract hyperparameters
    learning_rate = params['learning_rate']
    buffer_size = params['buffer_size']
    batch_size = params['batch_size']
    target_update_interval = params['target_update_interval']
    net_arch = params['net_arch']
    situation = 'merge_in-v0'
    # Create environment and train the DQN model
    env = make_vec_env(situation, n_envs=8, vec_env_cls=SubprocVecEnv)
    model = DQN("MlpPolicy", env, policy_kwargs=dict(net_arch=net_arch),
                learning_rate=learning_rate, buffer_size=buffer_size,
                batch_size=batch_size, target_update_interval=target_update_interval,
                verbose=0)

    model.learn(total_timesteps=int(2e4), log_interval=1000, progress_bar=False)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # Minimize the negative mean reward (maximize the reward)
    return -mean_reward
if __name__ == "__main__":
    freeze_support()
    # Define the hyperparameter search space for DQN
    space = {
        'learning_rate': hp.loguniform('learning_rate', -6, -3),
        'buffer_size': hp.choice('buffer_size', [5000, 10000, 20000]),
        'batch_size': hp.choice('batch_size', [16, 32, 64]),
        'target_update_interval': hp.choice('target_update_interval', [1000, 5000, 10000]),
        'net_arch': hp.choice('net_arch', [[64, 64], [128, 128], [256, 256]])
    }

    # Perform random search optimization for DQN
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10)

    print("Best hyperparameters for DQN:", best)