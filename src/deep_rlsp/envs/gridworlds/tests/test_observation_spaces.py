import unittest

import numpy as np

from deep_rlsp.envs.gridworlds import TOY_PROBLEMS
from deep_rlsp.envs.gridworlds.gym_envs import get_gym_gridworld


class TestObservationSpaces(unittest.TestCase):
    # def test_observation_state_consistency(self):
    #     for env_name, problems in TOY_PROBLEMS.items():
    #         for env_spec in problems.keys():
    #             _, cur_state, _, _ = TOY_PROBLEMS[env_name][env_spec]
    #             env = get_gym_gridworld(env_name, env_spec)
    #             for s in [env.init_state, cur_state]:
    #                 obs = env.s_to_obs(s)
    #                 s2 = env.obs_to_s(obs)
    #                 # if s != s2:
    #                 #     import pdb
    #                 #
    #                 #     pdb.set_trace()
    #                 self.assertEqual(s, s2)
    #
    # def test_no_error_getting_state_from_random_obs(self):
    #     np.random.seed(29)
    #     for env_name, problems in TOY_PROBLEMS.items():
    #         for env_spec in problems.keys():
    #             env = get_gym_gridworld(env_name, env_spec)
    #             for _ in range(100):
    #                 obs = np.random.random(env.obs_shape) * 4 - 2
    #                 state = env.obs_to_s(obs)
    #                 # if state not in env.state_num:
    #                 #     import pdb
    #                 #
    #                 #     pdb.set_trace()
    #                 self.assertIn(state, env.state_num)

    def test_obs_state_consistent_in_rollout(self):
        np.random.seed(29)
        for env_name, problems in TOY_PROBLEMS.items():
            for env_spec in problems.keys():
                env = get_gym_gridworld(env_name, env_spec)
                if not env.use_pixels_as_observations:
                    obs = env.reset()
                    done = False
                    while not done:
                        action = env.action_space.sample()
                        obs, reward, done, info = env.step(action)
                        s = env.obs_to_s(obs)
                        if s != env.s:
                            import pdb

                            pdb.set_trace()
                        self.assertEqual(s, env.s)


if __name__ == "__main__":
    unittest.main()
