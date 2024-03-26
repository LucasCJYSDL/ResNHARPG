import os
import numpy as np
import torch
import torch.optim as optim
from torch.multiprocessing import Pool
from tqdm import tqdm

from matplotlib import pyplot as plt
from itertools import repeat
from scipy.interpolate import Rbf
import scipy.stats as st

from worker import Worker
from utils import torch_load_cpu, get_inner_model, env_wrapper
from functions import euclidean_dist, FedPG_agg, MDA, CWTM, CWMed, MeaMed, Krum, GM, SimpleMean

class Memory:
    def __init__(self):
        self.steps = {} # progress shown as the number of trajectories collected
        self.eval_values = {} # average trajectory return when evaluating
        self.training_values = {} # average trajectory return when training

# def worker_run(worker, param, opts, Batch_size, seed, is_anpg=False):
def worker_run(worker, param, opts, Batch_size, seed, old_param, epoch):
    
    # distribute current parameters
    worker.load_param_from_master(param)
    worker.env.seed(seed)
    
    # get returned gradients and info from all agents
    # if is_anpg:
    #     out = worker.ANPG()

    if opts.ResNHARPG:
        old_worker = Worker(
                        id = -3,
                        is_Byzantine = False,
                        env_name = opts.env_name,
                        gamma = opts.gamma,
                        hidden_units = opts.hidden_units,
                        activation = opts.activation,
                        output_activation = opts.output_activation,
                        max_epi_len = opts.max_epi_len,
                        opts = opts).to(opts.device)
        old_worker.load_param_from_master(old_param)
        old_worker.env.seed(seed+1) #TODO: old worker and cur worker use different random seeds
        # TODO: define \eta_t in algorithm 2 based on t (i.e., epoch here)
        # eta = 2.0 / (epoch//100 + 2.0)
        eta = 0.99
        out = worker.train_one_epoch_HARPG(opts.device, opts.do_sample_for_training, old_worker, eta)

    else:
        out = worker.train_one_epoch(Batch_size, opts.device, opts.do_sample_for_training)

    # store all values
    return out
    

class Agent:
    
    def __init__(self, opts):
        # figure out the options
        self.opts = opts
        # setup arrays for distributed RL
        self.world_size = opts.num_worker
        # figure out the master
        self.master = Worker(
                id = 0,
                is_Byzantine = False,
                env_name = opts.env_name,
                gamma = opts.gamma,
                hidden_units = opts.hidden_units, 
                activation = opts.activation, 
                output_activation = opts.output_activation,
                max_epi_len = opts.max_epi_len,
                opts = opts
        ).to(opts.device)
        
        # figure out a copy of the master node for importance sampling purpose
        self.old_master = Worker(
                id = -1,
                is_Byzantine = False,
                env_name = opts.env_name,
                gamma = opts.gamma,
                hidden_units = opts.hidden_units, 
                activation = opts.activation, 
                output_activation = opts.output_activation,
                max_epi_len = opts.max_epi_len,
                opts = opts
        ).to(opts.device)

        # old_master for HARPG
        self.HARPG_old_master = Worker(
                id = -2,
                is_Byzantine = False,
                env_name = opts.env_name,
                gamma = opts.gamma,
                hidden_units = opts.hidden_units,
                activation = opts.activation,
                output_activation = opts.output_activation,
                max_epi_len = opts.max_epi_len,
                opts = opts
        ).to(opts.device)
        
        # figure out all the actors
        self.workers = []
        self.true_Byzantine = []
        for i in range(self.world_size): # so, in total N+2 workers
            self.true_Byzantine.append(True if i < opts.num_Byzantine else False)
            self.workers.append( Worker(
                                    id = i+1,
                                    is_Byzantine = True if i < opts.num_Byzantine else False,
                                    env_name = opts.env_name,
                                    gamma = opts.gamma,
                                    hidden_units = opts.hidden_units, 
                                    activation = opts.activation, 
                                    output_activation = opts.output_activation,
                                    attack_type =  opts.attack_type,
                                    max_epi_len = opts.max_epi_len,
                                    opts = opts
                            ).to(opts.device))
        print(f'{opts.num_worker} workers initilized with {opts.num_Byzantine if opts.num_Byzantine>0 else "None"} of them are Byzantine.')
        
        if not opts.eval_only:
            # figure out the optimizer TODO: use SGD
            # if opts.ResNHARPG or opts.ResPG:
            #     self.optimizer = optim.SGD(self.master.logits_net.parameters(), lr = opts.lr_model)
            # else:
            #     self.optimizer = optim.Adam(self.master.logits_net.parameters(), lr = opts.lr_model)

            self.optimizer = optim.Adam(self.master.logits_net.parameters(), lr=opts.lr_model)
        
        self.pool = Pool(self.world_size)
        self.memory = Memory()
    
    def load(self, load_path):
        assert load_path is not None
        load_data = torch_load_cpu(load_path)
        # load data for actor
        model_actor = get_inner_model(self.master.logits_net)
        model_actor.load_state_dict({**model_actor.state_dict(), **load_data.get('master', {})})
        
        if not self.opts.eval_only: # a rare case
            # load data for optimizer
            self.optimizer.load_state_dict(load_data['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.opts.device)
            # load data for torch and cuda
            torch.set_rng_state(load_data['rng_state'])
            if self.opts.use_cuda:
                torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
    
        print('[*] Loading data from {}'.format(load_path))
    
    def save(self, epoch, run_id):
        print('Saving model and state...')
        torch.save(
            {
                'master': get_inner_model(self.master.logits_net).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.opts.save_dir, 'r{}-epoch-{}.pt'.format(run_id,epoch))
        )
    
    
    def eval(self):
        # turn model to eval mode
        self.master.eval()
        
    def train(self):
        # turn model to trainig mode
        self.master.train()

    def _VR_update(self, b, N_t, opts, mu):

        for n in tqdm(range(N_t), desc='Master node'):
            # calculate new gradient in master node
            self.optimizer.zero_grad()

            # sample b trajectory using the latest policy (\theta_n) of master node
            weights, new_logp, batch_rets, batch_lens, batch_states, batch_actions = self.master.collect_experience_for_training(
                b,
                opts.device,
                record=True,
                sample=opts.do_sample_for_training)

            # calculate gradient for the new policy (\theta_n)
            loss_new = -(new_logp * weights).mean()
            self.master.logits_net.zero_grad()
            loss_new.backward()

            if mu:
                # get the old log_p with the old policy (\theta_0) but fixing the actions to be the same as the sampled trajectory
                old_logp = []
                for idx, obs in enumerate(batch_states):
                    # act in the environment with the fixed action
                    obs = env_wrapper(opts.env_name, obs)
                    _, old_log_prob = self.old_master.logits_net(
                        torch.as_tensor(obs, dtype=torch.float32).to(opts.device),
                        fixed_action=batch_actions[idx])  # TODO: improve by running in batches
                    # store in the old_logp
                    old_logp.append(old_log_prob)
                old_logp = torch.stack(old_logp)

                # Finding the ratio (pi_theta / pi_theta_old):
                # print(old_logp, new_logp)
                ratios = torch.exp(old_logp.detach() - new_logp.detach())  # important to detach

                # calculate gradient for the old policy (\theta_0)
                loss_old = -(old_logp * weights * ratios).mean()  # TODO: do not align with the formula
                self.old_master.logits_net.zero_grad()
                loss_old.backward()
                grad_old = [item.grad for item in self.old_master.parameters()]

                # early stop if ratio is not within [0.995, 1.005]
                if torch.abs(ratios.mean()) < 0.995 or torch.abs(ratios.mean()) > 1.005:
                    N_t = n
                    break

                # if tb_logger is not None:
                #     tb_logger.add_scalar(f'params/ratios_{run_id}', ratios.mean(), ratios_step)

                # adjust and set the gradient for latest policy (\theta_n)
                for idx, item in enumerate(self.master.parameters()):
                    item.grad = item.grad - grad_old[idx] + mu[idx]  # if mu is None, use grad from master
                    # grad_array += (item.grad.data.view(-1).cpu().tolist())

            # take a gradient step
            self.optimizer.step()
    
    def start_training(self, tb_logger = None, run_id = None):
        # run_id is the current seed id, also the repeated experiment id
        # parameters of running
        opts = self.opts

        # for storing number of trajectories sampled
        step = 0 # number of trajectories sampled
        epoch = 0 # training epoch
        
        # Start the training loop
        while step <= opts.max_trajectories:
            # step stands for the training iteration
            # epoch for storing checkpoints of model
            epoch += 1
            
            # Turn model into training mode
            print('\n\n')
            print("|",format(f" Training step {step} run_id {run_id} in {opts.seeds}","*^60"),"|")
            self.train()
            
            # setup lr_scheduler
            print("Training with lr={:.3e} for run {}".format(self.optimizer.param_groups[0]['lr'], opts.run_name) , flush=True)
            
            # to store rollout information from each worker, (n, )
            gradient = []
            batch_loss = []
            batch_rets = []
            batch_lens = []
            
            # distribute current params and Batch_Size to all workers
            old_param = get_inner_model(self.HARPG_old_master.logits_net).state_dict()
            param = get_inner_model(self.master.logits_net).state_dict()
            
            if opts.FedPG_BR:
                Batch_size = np.random.randint(opts.Bmin, opts.Bmax + 1)
            elif opts.ResNHARPG:
                Batch_size = 2 # TODO: only one trajectory is sampled from either policy in each epoch
            else:
                Batch_size = opts.B
        
            seeds = np.random.randint(1, 100000, self.world_size).tolist()

            # if opts.ResANPG:
            #     algo_id = [True for _ in range(self.world_size)]
            # else:
            #     algo_id = [False for _ in range(self.world_size)]

            args = zip(self.workers,
                       repeat(param),
                       repeat(opts),
                       repeat(Batch_size),
                       seeds,
                       repeat(old_param),
                       repeat(epoch))
            results = self.pool.starmap(worker_run, args) # broadcast the master parameter and collect gradients in parallel

            #  collect the gradient(for training), loss(for logging only), returns(for logging only), and epi_length(for logging only) from workers         
            for out in tqdm(results, desc='Worker node'):
                grad, loss, rets, lens = out #  grad is a list of gradients for each parameter in the net
                # store all values
                gradient.append(grad)
                batch_loss.append(loss)
                batch_rets.append(rets)
                batch_lens.append(lens)

            # simulate FedScsPG-attack (if needed) on server for demo, the bad workers collaborate with each other
            if opts.attack_type == 'FedScsPG-attack' and opts.num_Byzantine > 0:  
                for idx, _ in enumerate(self.master.parameters()):
                    tmp = []
                    for bad_worker in range(opts.num_Byzantine):
                        tmp.append(gradient[bad_worker][idx].view(-1))
                    tmp = torch.stack(tmp)

                    estimated_2sigma = euclidean_dist(tmp, tmp).max()
                    estimated_mean = tmp.mean(0)
                    
                    # change the gradient to be estimated_mean + 3sigma (with a random direction rnd)
                    rnd = torch.rand(gradient[0][idx].shape) * 2. - 1.
                    rnd = rnd / rnd.norm()
                    attacked_gradient = estimated_mean.view(gradient[bad_worker][idx].shape) + rnd * estimated_2sigma * 3. / 2.
                    for bad_worker in range(opts.num_Byzantine):
                        gradient[bad_worker][idx] = attacked_gradient
              
            # make the old policy as a copy of the current master node
            self.old_master.load_param_from_master(param)
            self.HARPG_old_master.load_param_from_master(param)
            
            # do Aggregate Algorithm to detect Byzantine worker on master node
            if opts.FedPG_BR:
                mu = FedPG_agg(self.old_master, self.world_size, gradient, opts, Batch_size)

            elif opts.ResNHARPG or opts.ResPG:
                agg_func = {'MDA': MDA, 'CWTM': CWTM, 'CWMed': CWMed, 'MeaMed': MeaMed, 'Krum': Krum, 'GM': GM,
                            'SimpleMean': SimpleMean}[opts.aggregator_name]
                if opts.aggregator_name == 'SimpleMean':
                    mu = agg_func(self.old_master, self.world_size, gradient, opts)
                else:
                    mu = agg_func(gradient, opts)
            
            # else will treat all nodes as non-Byzantine nodes
            else: #TODO: if we do not apply any aggregators to our algorithms, they will be the original PG algorithms
                mu = SimpleMean(self.old_master, self.world_size, gradient, opts)

            # perform gradient update in master node
            # grad_array = [] # store gradients for logging

            if opts.FedPG_BR or opts.SVRPG:
                # for n=1 to Nt ~ Geom(B/B+b) do grad update
                b = opts.b
                N_t = np.random.geometric(p=1 - Batch_size / (Batch_size + b))

                if opts.SVRPG:
                    b = opts.b
                    N_t = opts.N
                
                self._VR_update(b, N_t, opts, mu)

            elif opts.ResNHARPG:
                b = 0
                N_t = 0

                # TODO: apply the normalization to each layer or treat them as a whole
                # for idx, item in enumerate(self.master.parameters()):
                #     temp_norm = torch.norm(mu[idx], p='fro')
                #     item.grad = mu[idx] / temp_norm

                all_grads = []
                for i in range(len(mu)):
                    all_grads.append(mu[i].view(-1))
                #     print("1: ", mu[i].view(-1).shape)
                # print("2: ", torch.cat(all_grads).shape)
                all_norm = torch.norm(torch.cat(all_grads), p='fro')
                print("grad_norm: ", all_norm)
                for idx, item in enumerate(self.master.parameters()):
                    item.grad = mu[idx] / all_norm

                # TODO: adjust the learning rate for ResNHARPG in each epoch
                # cur_lr = 2.0 * 0.005 / (epoch + 2.0)
                if opts.env_name == 'CartPole-v1':
                    cur_lr = 0.005
                else:
                    cur_lr = 0.0005
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = cur_lr * 2.0 / (2.0 + epoch//1000) # 500 for Hopper, 1000 for DoublePendulum, 100 for others
                self.optimizer.step()
            else:
                b = 0
                N_t = 0
                # perform gradient descent with mu vector
                for idx, item in enumerate(self.master.parameters()):
                    item.grad = mu[idx]
                # take a gradient step
                self.optimizer.step()  
            
            print('\nepoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (epoch, np.mean(batch_loss), np.mean(batch_rets), np.mean(batch_lens)))
            
            # current step: number of trajectories sampled
            # step += max(Batch_size, b * N_t) if self.world_size > 1 else Batch_size + b * N_t
            step += round((Batch_size * self.world_size + b * N_t) / (1 + self.world_size)) if self.world_size > 1 else Batch_size + b * N_t
            
            # Logging to tensorboard, for training infos, these metrics are all 0 for ANPG, so only eval metrics are valid
            if(tb_logger is not None):
                # training log, these are the main metrics for the training process,
                tb_logger.add_scalar(f'train/total_rewards_{run_id}', np.mean(batch_rets), step)
                tb_logger.add_scalar(f'train/epi_length_{run_id}', np.mean(batch_lens), step)
                tb_logger.add_scalar(f'train/loss_{run_id}', np.mean(batch_loss), step)
                # grad log
                # tb_logger.add_scalar(f'grad/grad_{run_id}', np.mean(grad_array), step)
                # optimizer log
                tb_logger.add_scalar(f'params/lr_{run_id}', self.optimizer.param_groups[0]['lr'], step)
                # tb_logger.add_scalar(f'params/N_t_{run_id}', N_t, step)

                # for performance plot
                if run_id not in self.memory.steps.keys():
                    self.memory.steps[run_id] = []
                    self.memory.eval_values[run_id] = []
                    self.memory.training_values[run_id] = []
                
                self.memory.steps[run_id].append(step)
                self.memory.training_values[run_id].append(np.mean(batch_rets))

            # do validating
            eval_reward = self.start_validating(tb_logger, step, max_steps = opts.val_max_steps, render = opts.render, run_id = run_id)
            if(tb_logger is not None):
                 self.memory.eval_values[run_id].append(eval_reward)
                            
            # save current model
            if not opts.no_saving:
                self.save(epoch, run_id)
                
    
    # validate the new model   
    def start_validating(self, tb_logger = None, id = 0, max_steps = 1000, render = False, run_id = 0, mode = 'human'):
        # print('Validating...', flush=True)
        print('Validating...')
        # what are 'id' and 'run_id' for
        val_ret = 0.0
        val_len = 0.0
        
        for _ in range(self.opts.val_size):
            epi_ret, epi_len, _ = self.master.rollout(self.opts.device, max_steps = max_steps, render = render,
                                                      sample = False, mode = mode, save_dir = './outputs/',
                                                      filename = f'gym_{run_id}_{_}.gif') # sample is set as False
            val_ret += epi_ret
            val_len += epi_len
        
        val_ret /= self.opts.val_size
        val_len /= self.opts.val_size
        
        print('\nGradient step: %3d \t return: %.3f \t ep_len: %.3f'%(id, val_ret, val_len))
        
        if(tb_logger is not None): # this should be the main plot and metric
            tb_logger.add_scalar(f'validate/total_rewards_{run_id}', val_ret, id)
            tb_logger.add_scalar(f'validate/epi_length_{run_id}', val_len, id)
            # tb_logger.close()
        
        return val_ret
    
    # logging performance summary
    
    
    def plot_graph(self, array):
        plt.ioff()
        fig = plt.figure(figsize=(8,4))
        y = []
        
        for id in self.memory.steps.keys():
             x = self.memory.steps[id]
             y.append(Rbf(x, array[id], function = 'linear')(np.arange(self.opts.max_trajectories)))
        
        mean = np.mean(y, axis=0)
        
        l, h = st.norm.interval(0.90, loc=np.mean(y, axis = 0), scale=st.sem(y, axis = 0))
        
        plt.plot(mean)
        plt.fill_between(range(int(self.opts.max_trajectories)), l, h, alpha = 0.5)
        
        axes = plt.axes()
        axes.set_ylim([self.opts.min_reward, self.opts.max_reward])
        
        plt.xlabel("Number of Trajectories")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.tight_layout()
        return fig
    
    
    def log_performance(self, tb_logger):
       
        eval_img = self.plot_graph(self.memory.eval_values)
        training_img = self.plot_graph(self.memory.training_values)
        tb_logger.add_figure(f'validate/performance_until_{len(self.memory.steps.keys())}_runs', eval_img, len(self.memory.steps.keys()))
        tb_logger.add_figure(f'train/performance_until_{len(self.memory.steps.keys())}_runs', training_img, len(self.memory.steps.keys()))        
