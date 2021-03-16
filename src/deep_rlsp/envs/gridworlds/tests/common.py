import unittest

import numpy as np

from deep_rlsp.envs.gridworlds.env import Direction
from deep_rlsp.model import TabularTransitionModel


def get_directions():
    u, d, l, r, s = map(
        Direction.get_number_from_direction,
        [
            Direction.NORTH,
            Direction.SOUTH,
            Direction.WEST,
            Direction.EAST,
            Direction.STAY,
        ],
    )
    return u, d, l, r, s


def get_random_deterministic_policy(nS, nA):
    policy = np.zeros((nS, nA))
    for s in range(nS):
        a = np.random.randint(0, nA)
        policy[s, a] = 1
    return policy


def get_two_action_uniform_policy(nS, nA, a1, a2):
    policy = np.zeros((nS, nA))
    for s in range(nS):
        policy[s, a1] = 0.5
        policy[s, a2] = 0.5
    return policy


def get_trajectory_state_ids_from_policy(env, policy, T):
    state_ids = []
    env.reset()
    s = env.s
    for t in range(T):
        s_id = env.get_num_from_state(s)
        state_ids.append(s_id)
        a = np.random.choice(np.arange(env.nA), p=policy[s_id])
        s = env.state_step(a)
        env.s = s
    return state_ids


class BaseTests:
    # wrap in extra class s.t. the base class is not considered a test
    # see https://stackoverflow.com/a/25695512
    class TestEnv(unittest.TestCase):
        def check_trajectory(self, trajectory, reset=True):
            if reset:
                self.env.reset()
            state = self.env.s
            for action, (next_state, prob) in trajectory:
                # check state id consistency
                self.assertEqual(
                    self.env.get_state_from_num(self.env.get_num_from_state(state)),
                    state,
                )
                self.assertEqual(
                    self.env.get_num_from_state(
                        self.env.get_state_from_num(self.env.get_num_from_state(state))
                    ),
                    self.env.get_num_from_state(state),
                )

                actual_next_states = [
                    (s, p) for p, s, r in self.env.get_next_states(state, action)
                ]
                self.assertEqual(sum([p for _, p in actual_next_states]), 1.0)
                self.assertIn((next_state, prob), actual_next_states)

                # for deterministic environments check actual transition
                if self.env.is_deterministic():
                    self.assertEqual(self.env.state_step(action, state), next_state)
                    self.assertEqual(self.env.state_step(action), next_state)
                    features, reward, done, info = self.env.step(action)
                    self.assertEqual(self.env.s, next_state)

                state = next_state

        def test_trajectories(self):
            for trajectory in self.trajectories:
                self.check_trajectory(trajectory)

    class TestTabularTransitionModel(unittest.TestCase):
        def setUpDeterministic(self):
            """
            Add test cases to deterministic environments by:
                - selecting a random deterministic policy
                - rolling out the policy
                - setting the initial state of the environment to the first state of
                  the rollout
                - Test that for all transitions in the rollout, the transition model
                  predicts exactly this transition

            When everything is deterministic (transitions, policy and initial state),
            the trajectory is uniquely determined, and so given the timestep the
            (s, a, s') pair is uniquely determined. Thus, the transition model, which
            does get access to the timestep, should predict exactly this transition.
            """
            if self.env.is_deterministic():
                np.random.seed(1)
                for _ in range(5):
                    policy = get_random_deterministic_policy(self.env.nS, self.env.nA)
                    traj_state_ids = get_trajectory_state_ids_from_policy(
                        self.env, policy, 10
                    )
                    init_state_id = traj_state_ids[0]
                    transitions = []
                    for t, (last_s_id, s_id, next_s_id) in enumerate(
                        zip(
                            traj_state_ids[:-2],
                            traj_state_ids[1:-1],
                            traj_state_ids[2:],
                        )
                    ):
                        state = self.env.get_state_from_num(s_id)
                        probs_forward = np.zeros(self.env.nS)
                        probs_forward[next_s_id] = 1
                        probs_backward = np.zeros(self.env.nS)
                        probs_backward[last_s_id] = 1
                        transitions.append(
                            (
                                state,
                                t + 1,
                                probs_forward.tolist(),
                                probs_backward.tolist(),
                            )
                        )
                    initial_state_distribution = np.zeros(self.env.nS)
                    initial_state_distribution[init_state_id] = 1
                    self.model_tests.append(
                        {
                            "policy": policy,
                            "transitions": transitions,
                            "initial_state_distribution": initial_state_distribution,
                        }
                    )

        def check_single_step(
            self, transition_model, state, t, probs_forward, probs_backward
        ):
            probs_forward_model = transition_model.forward_step(
                state, return_probs_vector=True
            )
            self.assertEqual(np.sum(probs_forward_model), 1)
            self.assertTrue(np.allclose(np.array(probs_forward), probs_forward_model))

            forward_states_model = transition_model.forward_step(
                state, return_probs_vector=False
            )
            for s, p in forward_states_model:
                self.assertGreater(p, 0)
                self.assertLessEqual(p, 1)
                s_id = self.env.get_num_from_state(s)
                self.assertTrue(np.isclose(p, probs_forward[s_id]))

            probs_backward_model = transition_model.backward_step(
                state, t, return_probs_vector=True
            )
            self.assertEqual(np.sum(probs_backward_model), 1)
            self.assertTrue(np.allclose(np.array(probs_backward), probs_backward_model))

            backward_states_model = transition_model.backward_step(
                state, t, return_probs_vector=False
            )
            for s, p in backward_states_model:
                self.assertGreater(p, 0)
                self.assertLessEqual(p, 1)
                s_id = self.env.get_num_from_state(s)
                self.assertTrue(np.isclose(p, probs_backward[s_id]))

        def check_model_for_policy(self, model_test):
            transition_model = TabularTransitionModel(self.env)
            if "initial_state_distribution" in model_test:
                p0 = model_test["initial_state_distribution"]
            else:
                p0 = None
            transition_model.update_initial_state_distribution(p0)
            transition_model.update_policy(model_test["policy"])
            for state, t, probs_forward, probs_backward in model_test["transitions"]:
                self.check_single_step(
                    transition_model, state, t, probs_forward, probs_backward
                )

        def test_transition_model(self):
            for model_test in self.model_tests:
                self.check_model_for_policy(model_test)
