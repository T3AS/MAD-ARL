from __future__ import absolute_import, division, print_function

import argparse

import cv2
import gym
import macad_gym
import ray
from gym.spaces import Box, Discrete
from macad_agents.rllib.env_wrappers import wrap_deepmind
from macad_agents.rllib.models import register_mnih15_shared_weights_net
from ray import tune
from ray.rllib.agents.impala.vtrace_policy_graph import VTracePolicyGraph
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.tune import run_experiments
from ray.tune.registry import register_env

parser = argparse.ArgumentParser()

parser.add_argument("--num-iters", type=int, default=20)
parser.add_argument(
    "--num-workers",
    default=1,
    type=int,
    help="Num workers (CPU cores) to use")
parser.add_argument(
    "--num-gpus", default=1, type=int, help="Number of gpus to use. Default=2")
parser.add_argument(
    "--sample-bs-per-worker",
    default=50,
    type=int,
    help="Number of samples in a batch per worker. Default=50")
parser.add_argument(
    "--train-bs",
    default=500,
    type=int,
    help="Train batch size. Use as per available GPU mem. Default=500")
parser.add_argument(
    "--envs-per-worker",
    default=1,
    type=int,
    help="Number of env instances per worker. Default=10")
parser.add_argument(
    "--notes",
    default=None,
    help="Custom experiment description to be added to comet logs")

register_mnih15_shared_weights_net()
model_name = "mnih15_shared_weights"

env_name = "HomoNcomIndePOIntrxMASS3CTWN3-v0"
env = gym.make(env_name)
env_actor_configs = env.configs
num_framestack = env_actor_configs["env"]["framestack"]


def env_creator(env_config):
    import macad_gym
    env = gym.make("HomoNcomIndePOIntrxMASS3CTWN3-v0")
    # Apply wrappers to: convert to Grayscale, resize to 84 x 84,
    # stack frames & some more op
    env = wrap_deepmind(env, dim=84, num_framestack=num_framestack)
    return env


register_env(env_name, lambda config: env_creator(config))


# Placeholder to enable use of a custom pre-processor
class ImagePreproc(Preprocessor):
    def _init_shape(self, obs_space, options):
        self.shape = (84, 84, 3)  # Adjust third dim if stacking frames
        return self.shape

    def transform(self, observation):
        observation = cv2.resize(observation, (self.shape[0], self.shape[1]))
        return observation


ModelCatalog.register_custom_preprocessor("sq_im_84", ImagePreproc)

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    obs_space = Box(0.0, 255.0, shape=(84, 84, 3))
    act_space = Discrete(9)

    def gen_policy():
        config = {
            # Model and preprocessor options.
            "model": {
                "custom_model": model_name,
                "custom_options": {
                    # Custom notes for the experiment
                    "notes": {
                        "notes": args.notes
                    },
                },
                # NOTE:Wrappers are applied by RLlib if custom_preproc is NOT
                # specified
                "custom_preprocessor": "sq_im_84",
                "dim": 84,
                "free_log_std": False,  # if args.discrete_actions else True,
                "grayscale": True,
                # conv_filters to be used with the custom CNN model.
                # "conv_filters": [[16, [4, 4], 2], [32, [3, 3], 2],
                # [16, [3, 3], 2]]
            },
            # preproc_pref is ignored if custom_preproc is specified
            # "preprocessor_pref": "deepmind",

            # env_config to be passed to env_creator
            "env_config": env_actor_configs
        }
        return (VTracePolicyGraph, obs_space, act_space, config)

    policy_graphs = {
        a_id: gen_policy()
        for a_id in env_actor_configs["actors"].keys()
    }

    run_experiments({
        "MA-IMPALA-SSUI3CCARLA": {
            "run": "IMPALA",
            "env": env_name,
            "stop": {
                "training_iteration": args.num_iters
            },
            "config": {
                "log_level": "DEBUG",
                "num_sgd_iter": 10,
                "multiagent": {
                    "policy_graphs": policy_graphs,
                    "policy_mapping_fn":
                    tune.function(lambda agent_id: agent_id),
                },
                "num_workers": args.num_workers,
                "num_envs_per_worker": args.envs_per_worker,
                "sample_batch_size": args.sample_bs_per_worker,
                "train_batch_size": args.train_bs
            },
            "checkpoint_freq": 500,
            "checkpoint_at_end": True,
            "max_failures": 5
        }
    })
