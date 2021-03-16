import gym

from deep_rlsp.envs.gridworlds.one_hot_action_space_wrapper import (
    OneHotActionSpaceWrapper,
)
from deep_rlsp.envs.gridworlds import TOY_PROBLEMS, TOY_ENV_CLASSES


def get_gym_gridworld_id_time_limit(env_name, env_spec):
    id = env_name + "_" + env_spec
    id = "".join([s.capitalize() for s in id.split("_")])
    id += "-v0"
    spec, _, _, _ = TOY_PROBLEMS[env_name][env_spec]
    env = TOY_ENV_CLASSES[env_name](spec)
    return id, env.time_horizon


def make_gym_gridworld(env_name, env_spec):
    spec, _, _, _ = TOY_PROBLEMS[env_name][env_spec]
    env = TOY_ENV_CLASSES[env_name](spec)
    env = OneHotActionSpaceWrapper(env)
    return env


def get_gym_gridworld(env_name, env_spec):
    id, time_horizon = get_gym_gridworld_id_time_limit(env_name, env_spec)
    env = gym.make(id)
    return env
