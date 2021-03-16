from stable_baselines import SAC, PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as MlpPolicySac
from stable_baselines.sac.policies import FeedForwardPolicy as SACPolicy


class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, layers=[256, 256], feature_extraction="mlp")


def get_sac(env, **kwargs):
    env_id = env.unwrapped.spec.id
    if (
        env_id.startswith("Ant")
        or env_id.startswith("HalfCheetah")
        or env_id.startswith("Swimmer")
        or env_id.startswith("Fetch")
    ):
        sac_kwargs = {
            "verbose": 1,
            "learning_rate": 3e-4,
            "gamma": 0.98,
            "tau": 0.01,
            "ent_coef": "auto",
            "buffer_size": 1000000,
            "batch_size": 256,
            "learning_starts": 10000,
            "train_freq": 1,
            "gradient_steps": 1,
        }
        policy = CustomSACPolicy
    elif env_id.startswith("Hopper"):
        sac_kwargs = {
            "verbose": 1,
            "learning_rate": 3e-4,
            "ent_coef": 0.01,
            "buffer_size": 1000000,
            "batch_size": 256,
            "learning_starts": 1000,
            "train_freq": 1,
            "gradient_steps": 1,
        }
        policy = CustomSACPolicy
    else:
        sac_kwargs = {"verbose": 1, "learning_starts": 1000}
        policy = MlpPolicySac
    for key, val in kwargs.items():
        sac_kwargs[key] = val
    solver = SAC(policy, env, **sac_kwargs)
    return solver


def get_ppo(env, **kwargs):
    sac_kwargs = {"verbose": 1}
    policy = MlpPolicy
    for key, val in kwargs.items():
        sac_kwargs[key] = val
    solver = PPO2(policy, env, **sac_kwargs)
    return solver
