# Learning What To Do by Simulating the Past

This repository contains code that implements the _Deep Reward Learning by Simulating the Past (Deep RSLP)_ algorithm introduced in the paper ["Learning What To Do by Simulating the Past"](https://arxiv.org/abs/2104.03946). This code is provided as is, and will not be maintained. Here we describe how to reproduce the experimental results reported in the paper. You can find video of policies trained with Deep RLSP [here](https://sites.google.com/view/deep-rlsp).

### Citation

David Lindner, Rohin Shah, Pieter Abbeel, Anca Dragan. **Learning What To Do by Simulating the Past**. In _International Conference on Learning Representations (ICLR)_, 2021.

```
@inproceedings{lindner2021learning,
    title={Learning What To Do by Simulating the Past},
    author={Lindner, David and Shah, Rohin and Abbeel, Pieter and Dragan, Anca},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2021},
}
```

### Table of Contents

- [Set up the environment](#set-up-the-environment)
    - [Docker](#docker)
    - [Manual setup](#manual-setup)
- [Reproducing the experiments](#reproducing-the-experiments)
    - [Gridworld experiments](#gridworld-experiments)
    - [MuJoCo experiments](#mujoco-experiments)
- [Code quality](#code-quality)


## Set up the environment

There are two options to set up the environment to run the code: either using Docker, or setting up the environment manually using Anaconda. We recommend to use the Docker setup.

### Docker

You can use [Docker](https://www.docker.com/) to set up the dependencies including MuJoCo automatically. To do this install Docker, then copy a valid MuJoCo key to `docker/mjkey.txt`, and execute the following commands:

```
docker build --tag deep-rlsp:1.0 docker
docker run -v `pwd`:/deep-rlsp/ -w /deep-rlsp/ -i -t deep-rlsp:1.0 bash
conda activate deep-rlsp
```

The first command sets up a container with all required dependencies including MuJoCo. The second command starts an interactive shell inside the container and the third command activates the Anaconda environment set up inside the container. You can now run all experiments inside this container. Note, that you might have to modify `docker/Dockerfile` to use Tensorflow with GPU support.

### Manual setup

Alternatively, you can set up the same Anaconda environment manually. In this case [MuJoCo](http://www.mujoco.org/index.html) has to be installed locally. If using a non-standard location, the environment variables `MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MUJOCO_PATH` have to be set accordingly.

To perform the manual setup, install [Anaconda](https://www.anaconda.com/) locally and run the following commands to set up the environment:

```
conda env create -f docker/environment.yml
conda activate deep-rlsp
pip install mujoco-py==2.0.2.9
pip install -e .
conda activate deep-rlsp
```

This sets up an Anaconda environment with the required dependencies and activates it, which can then be used to run the code.


## Reproducing the experiments

Now we describe how to reproduce the experiments described in the paper. We first describe the experiments in Gridworld environments, discussed in Section 3.2, and then the experiments in MuJoCo environments, discussed in Sections 3.3 and 3.4. For each of these we describe how to run Deep RLSP, the ablations discussed in the paper, and GAIL as a baseline.

### Gridworld experiments

To run the Gridworld experiments reported in Section 3.2, you first have to train an inverse dynamics model for each environment:
```
python scripts/train_inverse_dynamics.py --gridworlds RoomDefault-v0
python scripts/train_inverse_dynamics.py --gridworlds ApplesDefault-v0
python scripts/train_inverse_dynamics.py --gridworlds TrainDefault-v0
python scripts/train_inverse_dynamics.py --gridworlds BatteriesDefault-v0
python scripts/train_inverse_dynamics.py --gridworlds BatteriesEasy-v0
python scripts/train_inverse_dynamics.py --gridworlds RoomBad-v0
```
The models will be saved in `tf_ckpt`, and will have names such as `tf_ckpt_mlp_RoomDefault-v0_20210313_160730`. You might have to create the folder `tf_ckpt` before running the models.

You can then run the experiments with the following commands:
```
python src/deep_rlsp/run.py with latent_rlsp_config room_default inverse_dynamics_model_checkpoint=tf_ckpt/tf_ckpt_vae_RoomDefault-v0_20200930_132218
python src/deep_rlsp/run.py with latent_rlsp_config train_default inverse_dynamics_model_checkpoint=tf_ckpt/tf_ckpt_vae_TrainDefault-v0_20200930_132234
python src/deep_rlsp/run.py with latent_rlsp_config apples_default inverse_dynamics_model_checkpoint=tf_ckpt/tf_ckpt_vae_ApplesDefault-v0_20200930_132414
python src/deep_rlsp/run.py with latent_rlsp_config batteries_easy inverse_dynamics_model_checkpoint=tf_ckpt/tf_ckpt_mlp_BatteriesDefault-v0_20200930_123401
python src/deep_rlsp/run.py with latent_rlsp_config batteries_default inverse_dynamics_model_checkpoint=tf_ckpt/tf_ckpt_mlp_BatteriesDefault-v0_20200930_123401
python src/deep_rlsp/run.py with latent_rlsp_config room_bad inverse_dynamics_model_checkpoint=tf_ckpt/tf_ckpt_vae_RoomDefault-v0_20200930_132218
```
adapting the paths of the inverse dynamics model.

You can run the "AverageFeatures" ablation by replacing `latent_rlsp_config` with `latent_rlsp_ablation` in the commands above.

### MuJoCo experiments

To reproduce our experiments in the MuJoCo simulator, that we report in Sections 3.3 and 3.4, you need to perform the following steps:

1. Obtain an original policy
2. Run Deep RLSP
3. Evaluate the results
4. Compare to baselines / ablations

We now describe each step in turn.

### Obtaining an original policy

We consider two different ways of obtaining policies to immitate:
1. Obtain policy by optimizing a given reward function
2. Obtain policy by running [Dynamics-Aware Unsupervised Discovery of Skills (DADS)](https://arxiv.org/abs/1907.01657)

#### Obtain policy by optimizing a given reward function

To train a policy on the reward function of a given MuJoCo environment, use the `scripts/train_sac.py` script. With the following commands you can train policies on the environments we discuss in the paper and save them in the `policies/` folder:

```
python scripts/train_sac.py InvertedPendulum-v2 policies/sac_pendulum_6e4 --timesteps 60000
python scripts/train_sac.py HalfCheetah-FW-v2 policies/sac_cheetah_fw_2e6 --timesteps 2000000
python scripts/train_sac.py HalfCheetah-BW-v2 policies/sac_cheetah_bw_2e6 --timesteps 2000000
python scripts/train_sac.py Hopper-v2 policies/sac_hopper_2e6 --timesteps 2000000
```

This uses the soft actor-critic algorithm to train a policy using the [hyperparameters from `rl-baselines-zoo`](https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/sac.yml). The hyperparameters are defined in `src/deep_rlsp/solvers/__init__.py`.

For convenience, we provide trained policies in the `policies/` folder of this repository.

#### Obtain policy by running DADS

We run DADS using the [code provided by the authors](https://github.com/google-research/dads). To reproduce the our experiments, we provide rollouts sampled from the `jumping` and `balancing` skills in the folder `skills/`.

### Run Deep RLSP

We are now ready to run the full Deep RLSP algorithm. The main file to run experiments is located at `src/deep_rlsp/run_mujoco.py`. The following commands reproduce the experiments discussed in the paper:

Pendulum
```
python src/deep_rlsp/run_mujoco.py with base pendulum n_sample_states=1
python src/deep_rlsp/run_mujoco.py with base pendulum n_sample_states=10
python src/deep_rlsp/run_mujoco.py with base pendulum n_sample_states=50
```

Cheetah running forward
```
python src/deep_rlsp/run_mujoco.py with base cheetah_fw n_sample_states=1
python src/deep_rlsp/run_mujoco.py with base cheetah_fw n_sample_states=10
python src/deep_rlsp/run_mujoco.py with base cheetah_fw n_sample_states=50

```

Cheetah running backward
```
python src/deep_rlsp/run_mujoco.py with base cheetah_bw n_sample_states=1
python src/deep_rlsp/run_mujoco.py with base cheetah_bw n_sample_states=10
python src/deep_rlsp/run_mujoco.py with base cheetah_bw n_sample_states=50
```

Hopper
```
python src/deep_rlsp/run_mujoco.py with base hopper n_sample_states=1
python src/deep_rlsp/run_mujoco.py with base hopper n_sample_states=10
python src/deep_rlsp/run_mujoco.py with base hopper n_sample_states=50
```

Cheetah balancing skill
```
python src/deep_rlsp/run_mujoco.py with base cheetah_skill current_state_file="skills/balancing_rollouts.pkl" n_sample_states=1
python src/deep_rlsp/run_mujoco.py with base cheetah_skill current_state_file="skills/balancing_rollouts.pkl" n_sample_states=10
python src/deep_rlsp/run_mujoco.py with base cheetah_skill current_state_file="skills/balancing_rollouts.pkl" n_sample_states=50
```

Cheetah jumping skill
```
python src/deep_rlsp/run_mujoco.py with base cheetah_skill current_state_file="skills/jumping_rollouts.pkl" n_sample_states=1
python src/deep_rlsp/run_mujoco.py with base cheetah_skill current_state_file="skills/jumping_rollouts.pkl" n_sample_states=10
python src/deep_rlsp/run_mujoco.py with base cheetah_skill current_state_file="skills/jumping_rollouts.pkl" n_sample_states=50
```

The results will be saved in the `results/` folder. The trained (VAE and dynamics) models will be saved in `tf_ckpt`.

### Evaluate the results

In the paper, we evaluate Deep RLSP in two ways:
1. Train a new policy on the inferred reward function from Deep RLSP and evaluate this policy (as in Table 1)
2. Evaluate the policy trained during Deep RLSP (for the balancing and jumping skills)

#### Train a new policy on the inferred reward function

To produce the results provided in Table 1 in the paper, we run SAC on the final reward function inferred by the Deep RLSP algorithm. To do this run the following command
```
python scripts/mujoco_evaluate_inferred_reward.py with experiment_folder=results/mujoco/20200528_150813_InvertedPendulum-v2_optimal
```
providing the subfolder of `results/` that corresponds to the experiment you want to evaluate. This creates a sub-folder in `results/mujoco/eval` that contains the trained policy.

Then, to evaluate this policy, run
```
python scripts/evaluate_policy.py results/mujoco/eval/20200605_203113_20200603_220928_InvertedPendulum-v2_optimal_1/policy.zip sac InvertedPendulum-v2 --num_rollouts 100
```
for the corresponding policy file. This samples 100 trajectories from the policy and determines the mean and standard deviation of the policy return.

The same script can also be used to visualize the policies using the `--render` or `--out_video` arguments.

#### Evaluate the policy trained during Deep RLSP

The policies trained during Deep RLSP are saved in the results folder of a specific run as `rlsp_policy_1.zip`, `rlsp_policy_2.zip`, ...

To evaluate these policies, run
```
python scripts/evaluate_policy.py results/mujoco/20200528_150813_InvertedPendulum-v2_optimal/rlsp_policy_112.zip sac InvertedPendulum-v2 --num_rollouts 100
```
for the corresponding policy file. This samples 100 trajectories from the policy and determines the mean and standard deviation of the policy return.

The same script can also be used to visualize the policies using the `--render` or `--out_video` arguments.


#### AverageFeatures and Waypoints ablations

To ensure comparability with a limited number of random seeds, we run the ablations with the same trained VAE and dynamics models and the same input states as Deep RLSP. This can be done the following commands:
```
python src/deep_rlsp/ablation_AverageFeatures.py with result_folder=results/mujoco/20200528_150813_InvertedPendulum-v2_optimal
python src/deep_rlsp/ablation_Waypoints.py with result_folder=results/mujoco/20200528_150813_InvertedPendulum-v2_optimal
```
passing a folder containing the corresponding results of Deep RLSP as an argument. The policy returned by this baseline algorithm can be found in `results/mujoco/`, and they can also be visualized and evaluated using the `scripts/evaluate_policy.py` script.

#### Compare to GAIL

Running  [Generative Adversarial Imitation Learning (GAIL)](https://arxiv.org/abs/1606.03476) requires the [_imitation_](https://github.com/HumanCompatibleAI/imitation) library. You can install it using:
```
pip install imitation==0.1.1
```

To run GAIL, we provide demonstrations from the expert policies in the correct format in the `demonstrations` folder. You can create demonstration data from already trained expert policies by running:
```
python scripts/create_demonstrations.py policies/sac_cheetah_fw_2e6.zip demonstrations/sac_cheetah_fw_traj_len_{}_seed_{}.pkl 10 generate_seed HalfCheetah-FW-v2 1
```

Then you can run GAIL on the demonstration data by running:
```
python scripts/run_gail.py with gail half_cheetah env_name='HalfCheetah-FW-v2' rollout_path=demonstrations/sac_cheetah_fw_traj_len_1_seed_22750069.pkl log_dir=./gail_logs/gail_cheetah_fw_len_1_demoseed_22750069/
```

To visualize the resulting policies:
```
python scripts/evaluate_policy.py gail_logs/gail_cheetah_fw_len_1_demoseed_22750069/checkpoints/final/gen_policy gail HalfCheetah-FW-v2 --render --out_video=videos/gail_balancing_len_1.mp4
```

## Code quality

We use `black` for code formatting, `flake8` for linting, and `mypy` to check type hints.
You can run all checks with `bash code_checks.sh` and unit tests
with `python setup.py test`.
