from rllab.envs.gym_env import GymEnv
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.envs.base import Env
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy


import pickle
import os.path as osp
import numpy as np
from baselines.utils import *

import tensorflow as tf


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("env", default=None, help="The list of environments to train on in order. Eval rollouts will be run on all environments at the end.")
parser.add_argument("--num_epochs", default=10000, type=int, help="Number of epochs to run.")
parser.add_argument("--num_final_rollouts", default=20, type=int, help="Number of rollouts to run on final evaluation of environments.")
parser.add_argument("--batch_size", default=25000, type=int, help="Batch_size per epoch (this is the number of (state, action) samples, not the number of rollouts)")
parser.add_argument("--step_size", default=0.01, type=float, help="Step size for TRPO (i.e. the maximum KL bound)")
parser.add_argument("--reg_coeff", default=1e-5, type=float, help="Regularization coefficient for TRPO's conjugate gradient")
parser.add_argument("--text_log_file", default="./data/debug.log", help="Where text output will go")
parser.add_argument("--tabular_log_file", default="./data/progress.csv", help="Where tabular output will go")

args = parser.parse_args()

# stub(globals())

# ext.set_seed(1)
logger.add_text_output(args.text_log_file)
logger.add_tabular_output(args.tabular_log_file)
logger.set_log_tabular_only(False)


def run_task(*_):
    # Non-registration of this custom environment is an rllab bug
    # See https://github.com/openai/rllab/issues/68 
    # At the moment I'm bypassing this problem by adding the
    # import statement in gym_env.py
    import gym_follower_2d
    import lasagne.nonlinearities as NL
        
    gymenv = GymEnv(args.env, force_reset=True, record_video=False,  record_log=True)
    env = normalize(gymenv)
    
    logger.log("Training Policy on %s" % args.env)
 
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25),
        hidden_nonlinearity=NL.tanh
    )
    
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=args.batch_size,
        max_path_length=100,
        n_itr=args.num_epochs,
        discount=0.99,
        step_size=args.step_size,
        optimizer=ConjugateGradientOptimizer(reg_coeff=args.reg_coeff, hvp_approach=FiniteDifferenceHvp(base_eps=args.reg_coeff)),
        plot=False,
    )

    algo.train()
                        


run_experiment_lite(
    run_task,
    n_parallel=3,
    snapshot_mode="last",
    seed=1,
    plot=False,
    use_gpu=False,
)




