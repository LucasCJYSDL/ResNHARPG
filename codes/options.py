#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import torch

def get_options(args=None):
    parser = argparse.ArgumentParser('FPGA')

    ### Overall run settings
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1', choices = ['CartPole-v1', 'InvertedPendulum-v2'],
                        help='OpenAI Gym env name for test') # *
    parser.add_argument('--eval_only', action='store_true', 
                        help='used only if to evaluate a pre-trained model')
    parser.add_argument('--no_saving', action='store_true', 
                        help='Disable saving checkpoints')
    parser.add_argument('--no_tb', action='store_true', 
                        help='Disable Tensorboard logging')
    parser.add_argument('--render', action='store_true', 
                        help='render to view the env')
    parser.add_argument('--mode', type=str, choices = ['human', 'rgb'], default='human', 
                        help='render mode')
    parser.add_argument('--log_dir', default = 'logs', 
                        help='log folder' )
    parser.add_argument('--run_name', default='run_name', 
                        help='Name to identify the experiments') # name for the training session

    # I suppose the log_dir and run_name need not to be specified everytime
    
    # Multiple runs
    parser.add_argument('--multiple_run', type=int, default=5,
                        help='number of repeated runs') # *
    parser.add_argument('--seed', type=int, default=0, 
                        help='Starting point of random seed when running multiple times') # *

    # Federation and Byzantine parameters
    parser.add_argument('--num_worker', type=int, default=10, 
                        help = 'number of worker node') # *
    parser.add_argument('--num_Byzantine', type=int, default=3,
                        help = 'number of worker node that is Byzantine') # *
    parser.add_argument('--alpha', type=float, default=0.4, 
                        help = 'at most alpha-fractional worker nodes are Byzantine') # *
    parser.add_argument('--attack_type', type=str, default='random-noise',
                        choices = ['zero-gradient', 'random-action', 'sign-flipping',
                                   'reward-flipping', 'random-noise', 'FedScsPG-attack'],
                        help = 'the behavior scheme of a Byzantine worker') # *

    # RL Algorithms (default GOMDP) # three baselines
    parser.add_argument('--GOMDP', action='store_true', # nothing
                        help='run GOMDP')
    parser.add_argument('--SVRPG', action='store_true', # VR
                        help='run SVRPG')
    parser.add_argument('--FedPG_BR', action='store_true', # VR + aggregator
                        help='run FT-FedScsPG')

    ## ours
    parser.add_argument('--ResPG', action='store_true', # algorithm 1, aggregator
                        help='run ResPG')
    # parser.add_argument('--ResANPG', action='store_true',
    #                     help='run ResANPG') # NPG
    parser.add_argument('--ResNHARPG', action='store_true', help='run ResNHARPG') # algorithm 2, aggregator + hessian + norm

    parser.add_argument('--aggregator_name', '--agg', type=str, default='MDA',
                        choices=['MDA', 'CWTM', 'CWMed', 'MeaMed', 'Krum', 'GM', 'SimpleMean'],
                        help='Aggregator functions being adopted.')  # 6 aggregators

    # each neural network layer contains w and b, should we aggregate each layer independently or as a whole (concatenation)?
    parser.add_argument('--compute_per_component', action='store_true',
                        help='apply the aggregators to each parameter module or view them as a whole')

    # # ANPG related hyperparameters
    # parser.add_argument('--alpha_npg', type=float, default=0.5, help='alpha in ANPG')  # *
    # parser.add_argument('--beta_npg', type=float, default=0.5, help='beta in ANPG')  # *
    # parser.add_argument('--delta_npg', type=float, default=0.5, help='delta in ANPG')  # *
    # parser.add_argument('--xi_npg', type=float, default=0.5, help='xi in ANPG')  # *
    # parser.add_argument('--H_npg', type=float, default=10, help='H in ANPG')  # *
    
    # Training and validating
    parser.add_argument('--val_size', type=int, default=10, 
                        help='Number of episodes used for reporting validation performance')
    parser.add_argument('--val_max_steps', type=int, default=1000, 
                        help='Maximum trajectory length used for reporting validation performance')

    # Load pre-trained modelss
    parser.add_argument('--load_path', default = None,
                        help='Path to load pre-trained model parameters')

    ### end of parameters
    opts = parser.parse_args(args)

    if opts.SVRPG:
        alg_name = 'SVRPG'
    elif opts.GOMDP:
        alg_name = 'GPMDP'
    elif opts.FedPG_BR:
        alg_name = 'FT-FedScsPG'
    elif opts.ResNHARPG:
        alg_name = 'ResNHARPG'
    else:
        assert opts.ResPG
        alg_name = 'ResPG'

    print("run {}!!!!!!!!!!!!!!!!!!!!".format(alg_name))

    opts.use_cuda = False # danger
    opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    opts.save_dir = os.path.join(
        'outputs',
        '{}'.format(opts.env_name),
        "{}_{}_worker{}_byzantine{}_{}".format(alg_name, opts.aggregator_name, opts.num_worker, opts.num_Byzantine, opts.attack_type),
        opts.run_name
    ) if not opts.no_saving else None
    opts.log_dir = os.path.join(
        f'{opts.log_dir}',
        '{}'.format(opts.env_name),
        "{}_{}_worker{}_byzantine{}_{}".format(alg_name, opts.aggregator_name, opts.num_worker, opts.num_Byzantine, opts.attack_type),
        opts.run_name
    ) if not opts.no_tb else None
    
    if opts.env_name == 'CartPole-v1':
        # Task-Specified Hyperparameters
        opts.max_epi_len = 500
        opts.max_trajectories = 1800 # 5000
        opts.gamma = 0.999
        opts.min_reward = 0  # for logging purpose (not important)
        opts.max_reward = 600  # for logging purpose (not important)

        # shared parameters
        opts.do_sample_for_training = True
        opts.lr_model = 1e-3
        opts.hidden_units = '16,16'
        opts.activation = 'ReLU'
        opts.output_activation = 'Tanh'
        
        # batch_size
        opts.B = 16
        opts.Bmin = 12 # for FT-FedScsPG
        opts.Bmax = 20 # for FT-FedScsPG
        opts.b = 4 # mini batch_size for VR
        opts.N = 3 # inner loop iteration for VR
        
        # Filtering hyperparameters for FT-FedScsPG
        opts.delta = 0.6
        opts.sigma = 0.06

  
    elif opts.env_name == 'InvertedPendulum-v2':
        # Task-Specified Hyperparameters
        opts.max_epi_len = 500  
        opts.max_trajectories = 1800
        opts.gamma  = 0.995
        opts.min_reward = 0 # for logging purpose (not important)
        opts.max_reward = 1200 # for logging purpose (not important)
        
        # shared parameters
        opts.do_sample_for_training = True
        opts.lr_model = 8e-5 # 4e-3
        opts.hidden_units = '64,64'
        opts.activation = 'Tanh'
        opts.output_activation = 'Tanh'
       
        # batch_size
        opts.B = 48 # for SVRPG and GOMDP
        opts.Bmin = 46 # for FT-FedScsPG
        opts.Bmax = 50 # for FT-FedScsPG
        opts.b = 16 # mini batch_size for SVRPG and FT-FedScsPG
        opts.N = 3 # inner loop iteration for SVRPG
    
        # Filtering hyperparameters for FT-FedScsPG
        opts.delta = 0.6
        opts.sigma = 0.9

    else:
        raise NotImplementedError
    
    return opts
