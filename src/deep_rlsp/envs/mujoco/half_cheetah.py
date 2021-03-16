# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from gym import utils
import numpy as np
from gym.envs.mujoco import mujoco_env


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        expose_all_qpos=False,
        task="default",
        target_velocity=None,
        model_path="half_cheetah.xml",
        plot=False,
    ):
        # Settings from
        # https://github.com/openai/gym/blob/master/gym/envs/__init__.py
        self._expose_all_qpos = expose_all_qpos
        self._task = task
        self._target_velocity = target_velocity

        xml_path = os.path.join(os.path.dirname(__file__), "assets")
        self.model_path = os.path.abspath(os.path.join(xml_path, model_path))

        mujoco_env.MujocoEnv.__init__(self, self.model_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        xvelafter = self.sim.data.qvel[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()

        if self._task == "default":
            reward_vel = 0.0
            reward_run = (xposafter - xposbefore) / self.dt
            reward = reward_ctrl + reward_run
        elif self._task == "target_velocity":
            reward_vel = -((self._target_velocity - xvelafter) ** 2)
            reward = reward_ctrl + reward_vel
        elif self._task == "run_back":
            reward_vel = 0.0
            reward_run = (xposbefore - xposafter) / self.dt
            reward = reward_ctrl + reward_run

        done = False
        return (
            ob,
            reward,
            done,
            dict(reward_run=reward_run, reward_ctrl=reward_ctrl, reward_vel=reward_vel),
        )

    def _get_obs(self):
        if self._expose_all_qpos:
            return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])
        return np.concatenate([self.sim.data.qpos.flat[1:], self.sim.data.qvel.flat])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.sim.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.sim.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        # camera_id = self.model.camera_name2id("track")
        # self.viewer.cam.type = 2
        # self.viewer.cam.fixedcamid = camera_id
        # self.viewer.cam.distance = self.model.stat.extent * 0.5
        # camera_id = self.model.camera_name2id("fixed")
        # self.viewer.cam.type = 2
        # self.viewer.cam.fixedcamid = camera_id

        self.viewer.cam.fixedcamid = -1
        self.viewer.cam.trackbodyid = -1
        # how much you "zoom in", model.stat.extent is the max limits of the arena
        self.viewer.cam.distance = self.model.stat.extent * 0.4
        # self.viewer.cam.lookat[0] -= 4
        self.viewer.cam.lookat[1] += 1.1
        self.viewer.cam.lookat[2] += 0
        # camera rotation around the axis in the plane going through the frame origin
        # (if 0 you just see a line)
        self.viewer.cam.elevation = -20
        # camera rotation around the camera's vertical axis
        self.viewer.cam.azimuth = 90
