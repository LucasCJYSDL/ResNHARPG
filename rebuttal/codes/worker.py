import torch
import numpy as np
import gym
from gym.spaces import Discrete
from policy import MlpPolicy, DiagonalGaussianMlpPolicy
from utils import get_inner_model, save_frames_as_gif
from utils import env_wrapper
import random

class Worker:

    def __init__(self,
                 id,
                 is_Byzantine,
                 env_name,
                 hidden_units,
                 gamma,
                 activation = 'Tanh',
                 output_activation = 'Identity',
                 attack_type = None,
                 max_epi_len = 0,
                 opts = None
                 ):
        super(Worker, self).__init__()
        
        # setup
        self.id = id
        self.old_d = None # TODO: main the old_d for each worker?
        self.is_Byzantine = is_Byzantine
        self.gamma = gamma
        # make environment, check spaces, get obs / act dims
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.attack_type = attack_type
        self.max_epi_len = max_epi_len
        self.opts = opts
        
        assert opts is not None
        
        # get observation dim
        obs_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, Discrete):
            n_acts = self.env.action_space.n
        else:
            n_acts = self.env.action_space.shape[0]
        
        hidden_sizes = list(eval(hidden_units))
        self.sizes = [obs_dim]+hidden_sizes+[n_acts] # make core of policy network
        
        # get policy net
        if isinstance(self.env.action_space, Discrete):
            self.logits_net = MlpPolicy(self.sizes, activation, output_activation)
            # so the output is applied with tanh, which changes with tasks
        else:
            self.logits_net = DiagonalGaussianMlpPolicy(self.sizes, activation,)
        
        if self.id == 1:
            print(self.logits_net)

    
    def load_param_from_master(self, param):
        model_actor = get_inner_model(self.logits_net)
        model_actor.load_state_dict({**model_actor.state_dict(), **param}) # TODO: check

    def rollout(self, device, max_steps = 1000, render = False, env = None, obs = None,
                sample = True, mode = 'human', save_dir = './', filename = '.'):
        
        if env is None and obs is None:
            env = self.env
            obs = env.reset()
            
        done = False  
        ep_rew = []
        frames = []
        step = 0
        while not done and step < max_steps:
            step += 1
            if render:
                if mode == 'rgb':
                    frames.append(env.render(mode="rgb_array"))
                else:
                    env.render()
                
            obs = env_wrapper(env.unwrapped.spec.id, obs)
            action = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample)[0]
            obs, rew, done, _ = env.step(action)
            ep_rew.append(rew)

        if mode == 'rgb': save_frames_as_gif(frames, save_dir, filename)
        return np.sum(ep_rew), len(ep_rew), ep_rew
    
    def collect_experience_for_training(self, B, device, record = False, sample = True, attack_type = None):
        # make some empty lists for logging. For each of the B trajectories
        batch_weights = []      # for R(tau) weighting in policy gradient, return list of each trajectory
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        batch_log_prob = []     # for gradient computing

        # reset episode-specific variables
        obs = self.env.reset()  # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep, for one episode
        
        # make two lists for recording the trajectory
        batch_states = []
        batch_actions = []

        t = 1
        # collect experience by acting in the environment with current policy
        while True:
            # save trajectory
            if record:
                batch_states.append(obs)
            # act in the environment  
            obs = env_wrapper(self.env_name, obs)
            
            # TODO: simulate random-action attacker if needed
            if self.is_Byzantine and attack_type is not None and self.attack_type == 'random-action':
                act_rnd = self.env.action_space.sample()
                if isinstance(act_rnd, int): # discrete action space
                    act_rnd = 0
                else: # continuous
                    act_rnd = np.zeros(len(self.env.action_space.sample()), dtype=np.float32)  # this is not random action
                act, log_prob = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample, fixed_action = act_rnd)
            else:
                act, log_prob = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample)
           
            obs, rew, done, info = self.env.step(act)
            
            # simulate reward-flipping attacker if needed
            if self.is_Byzantine and attack_type is not None and self.attack_type == 'reward-flipping': 
                rew = - rew
                
            # timestep
            t = t + 1
            
            # save action_log_prob, reward
            batch_log_prob.append(log_prob)
            ep_rews.append(rew)
            
            # save trajectory
            if record:
                batch_actions.append(act)

            if done or len(ep_rews) >= self.max_epi_len:
                
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # so s_H is not included
                # the weight for each logprob(a_t|s_T) is sum_t^T (gamma^(t'-t) * r_t')
                returns = []
                R = 0
                # simulate random-reware attacker if needed
                # TODO: unify the way to calculate g for all algorithms
                if self.opts.ResNHARPG or self.opts.ResPG:
                    H = len(ep_rews)
                    for r_idx in range(H-1, -1, -1):
                        R += (self.gamma ** (r_idx)) * ep_rews[r_idx]
                        returns.insert(0, R) # TODO: the episode length for each trajectory can be different
                else:
                    for r in ep_rews[::-1]:
                        R = r + self.gamma * R
                        returns.insert(0, R)
                returns = torch.tensor(returns, dtype=torch.float32)
                
                # return whitening
                # TODO: whether to keep this
                advantage = (returns - returns.mean()) / (returns.std() + 1e-20)
                batch_weights += advantage # batch_weights is always 1-D, the same as logp

                # end experience loop if we have enough of it
                if len(batch_lens) >= B:
                    break
                
                # reset episode-specific variables
                obs, done, ep_rews, t = self.env.reset(), False, [], 1

        # make torch tensor and restrict to batch_size
        weights = torch.as_tensor(batch_weights, dtype = torch.float32).to(device) # 1-D torch tensor
        logp = torch.stack(batch_log_prob) # 1-D torch tensor

        if record:
            return weights, logp, batch_rets, batch_lens, batch_states, batch_actions
        else:
            return weights, logp, batch_rets, batch_lens
    
    
    def train_one_epoch(self, B, device, sample):
        
        # collect experience by acting in the environment with current policy
        weights, logp, batch_rets, batch_lens = self.collect_experience_for_training(B, device, sample=sample,
                                                                                     attack_type=self.attack_type)
        # weights: list of weights for each logp, epi_num * epi_len
        # batch_rets, batch_lens: summary for each trajectory, epi_num
        
        # calculate policy gradient loss
        batch_loss = -(logp * weights).mean()
    
        # take a single policy gradient update step
        self.logits_net.zero_grad()
        batch_loss.backward()

        # get grad from each worker
        # determine if the agent is byzantine
        if self.is_Byzantine and self.attack_type is not None:
            # return wrong gradient with noise
            grad = [] # the order of items in self.parameters() should be followed
            for item in self.parameters():
                if self.attack_type == 'zero-gradient':
                    grad.append(item.grad * 0)
                
                elif self.attack_type == 'random-noise':
                    rnd = (torch.rand(item.grad.shape, device = item.device) * 2 - 1) * (item.grad.max().data - item.grad.min().data) * 3
                    grad.append(item.grad + rnd)
                
                elif self.attack_type == 'sign-flipping':
                    grad.append(-2.5 * item.grad)
                    
                elif self.attack_type == 'reward-flipping':
                    grad.append(item.grad)
                    # refer to collect_experience_for_training() to see attack

                elif self.attack_type == 'random-action':
                    grad.append(item.grad)
                    # refer to collect_experience_for_training() to see attack
                
                elif self.attack_type == 'FedScsPG-attack':
                    grad.append(item.grad)
                    # refer to agent.py to see attack
                    
                else:
                    raise NotImplementedError()

    
        else:
            # return true gradient
            grad = [item.grad for item in self.parameters()]
        
        # report the results to the agent for training purpose
        return grad, batch_loss.item(), np.mean(batch_rets), np.mean(batch_lens) # the last three values are for evaluation only


    def train_one_epoch_HARPG(self, device, sample, old_worker, eta):

        q = np.random.rand()
        param = self.logits_net.state_dict()
        old_param = old_worker.logits_net.state_dict()
        u_list = []
        for name, v in old_param.items():
            u_list.append((param[name] - v).detach().clone().view(-1))
            old_param[name] = v * (1-q) + param[name] * q
        u = torch.cat(u_list, dim=0) # u do not contain gradient info
        # print("1: ", u, u.shape)
        old_worker.logits_net.load_state_dict(old_param)

        # tau
        weights, logp, batch_rets, batch_lens = self.collect_experience_for_training(1, device, sample=sample,
                                                                                     attack_type=self.attack_type)
        batch_loss = (logp * weights).sum() #TODO: use mean or not
        self.logits_net.zero_grad()
        batch_loss.backward()

        g_tau = []
        for item in self.parameters():
            g_tau.append(item.grad.view(-1))
        g_tau = torch.cat(g_tau, dim=0) # contain no gradient info
        # print("2: ", g_tau.shape, weights, logp, g_tau) # (386, )

        # hat_tau
        h_weights, h_logp, h_batch_rets, h_batch_lens = old_worker.collect_experience_for_training(1, device, sample=sample,
                                                                                     attack_type=self.attack_type)
        # get \nable_\log_p_\tau_\pi
        entropy_loss = h_logp.sum()
        old_worker.logits_net.zero_grad()
        entropy_loss.backward(retain_graph=True)
        nabla_logp_tau_pi = []
        for item in old_worker.parameters():
            nabla_logp_tau_pi.append(item.grad.view(-1))
        nabla_logp_tau_pi = torch.cat(nabla_logp_tau_pi, dim=0)

        # get other gradients
        h_batch_loss = (h_logp * h_weights).sum() #log_p has gradient info #TODO: use mean or not
        old_worker.logits_net.zero_grad()
        h_batch_loss.backward(create_graph=True)

        g_h_tau_list = []
        for item in old_worker.parameters():
            g_h_tau_list.append(item.grad.view(-1))
        g_h_tau = torch.cat(g_h_tau_list, dim=0).detach().clone() # saved for later, do not contain gradient info
        g_h_tau_2 = torch.cat(g_h_tau_list, dim=0).clone()
        # print("3: ", g_h_tau_2, g_h_tau.shape)

        # second order gradient
        second_order_loss = (g_h_tau_2 * u).sum()
        # print("4: ", (g_h_tau_2 * u).sum(), (g_h_tau_2 * u).shape)
        # zero out the old gradient
        for item in old_worker.parameters():
            item.grad.data.zero_()

        second_order_loss.backward()
        # get the second term of B*u
        nabla_g_u = []
        for item in old_worker.parameters():
            nabla_g_u.append(item.grad.view(-1))
        nabla_g_u = torch.cat(nabla_g_u, dim=0).detach().clone()
        # print("5:", nabla_g_u, nabla_g_u.shape)

        # get v
        v = (nabla_logp_tau_pi * u).sum() * g_h_tau + nabla_g_u
        # print("6: ", v, v.shape)

        if self.old_d is None:
            # TODO: how to initialize d?
            self.old_d = torch.zeros_like(v)

        # get d
        d = (1 - eta) * (self.old_d + v) + eta * g_tau
        # print("7: ", d, d.shape)

        # byzantine
        if self.is_Byzantine and self.attack_type is not None:
            if self.attack_type == 'zero-gradient':
                d = d * 0

            elif self.attack_type == 'random-noise':
                rnd = (torch.rand(d.shape, device=d.device) * 2 - 1) * (
                        d.max().data - d.min().data) * 3
                d = d + rnd

            elif self.attack_type == 'sign-flipping':
                d = -2.5 * d

        grad = []
        s_id = 0
        for item in self.parameters():
            e_id = s_id + int(item.grad.view(-1).shape[0])
            temp_grad = -1.0 * d[s_id:e_id].view(item.grad.shape) # the outer optimizer is gradient descent, so we reverse the sign here
            # print("8: ", temp_grad.shape)

            grad.append(temp_grad)

            s_id = e_id

        self.old_d = d.detach().clone()

        return grad, batch_loss.item(), np.mean(batch_rets), np.mean(batch_lens)


    # the following functions are for ANPG

    # def _take_action(self, obs):
    #     obs = env_wrapper(self.env_name, obs)
    #     # simulate random-action attacker if needed
    #     if self.is_Byzantine and self.attack_type == 'random-action':
    #         act_rnd = self.env.action_space.sample()
    #         if isinstance(act_rnd, int):  # discrete action space
    #             act_rnd = 0
    #         else:  # continuous
    #             act_rnd = np.zeros(len(self.env.action_space.sample()),
    #                                dtype=np.float32)  # this is not random action
    #         act, log_prob = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(self.opts.device),
    #                                         fixed_action=act_rnd)
    #     else:
    #         act, log_prob = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(self.opts.device))
    #
    #     return act, log_prob
    #
    # def _ANPG_sampling(self, w):
    #
    #     T = np.random.geometric(p=1-self.opts.gamma)
    #
    #     # sample 1
    #     obs = self.env.reset()
    #     done = False
    #     t = 0
    #     while (not done) and t < T: # TODO: deal with the case when done happens
    #         # act in the environment
    #         act, log_prob = self._take_action(obs)
    #         obs, rew, done, info = self.env.step(act)
    #
    #         # timestep
    #         t = t + 1
    #
    #     final_obs = obs
    #     final_act, final_logp = self._take_action(final_obs)
    #
    #     # print("4: ", final_logp)
    #
    #     # prepare for the next round sampling
    #     obs = final_obs
    #     X = np.random.binomial(n=1, p=0.5)
    #     T = np.random.geometric(p=1 - self.opts.gamma)
    #     if X == 1:
    #         act, _ = self._take_action(obs)
    #     else:
    #         act = final_act
    #
    #     # sample 2
    #     done = False
    #     t = 0
    #     ret = 0.0
    #     while done and t < T:
    #         obs, rew, done, info = self.env.step(act)
    #
    #         # simulate reward-flipping attacker if needed
    #         if self.is_Byzantine and self.attack_type == 'reward-flipping':
    #             rew = - rew
    #
    #         ret += rew
    #         act, _ = self._take_action(obs)
    #         t += 1
    #
    #     _, rew, _, _ = self.env.step(act)
    #
    #     # simulate reward-flipping attacker if needed
    #     if self.is_Byzantine and self.attack_type == 'reward-flipping':
    #         rew = - rew
    #
    #     ret += rew
    #
    #     # final
    #     Q_hat = 2.0 * (1-X) * ret
    #     V_hat = 2.0 * X * ret
    #     A_hat = Q_hat - V_hat
    #
    #     batch_loss = (final_logp).mean() # TODO: whether to be negative
    #     self.logits_net.zero_grad()
    #     batch_loss.backward()
    #
    #     grad = [item.grad for item in self.parameters()] # list of tensors
    #
    #     mu_vec = None
    #     for idx in range(len(grad)):
    #         grad_item = grad[idx].view(-1)
    #         # print("1: ", grad_item.shape)
    #         # concat stacked grad vector
    #         if mu_vec is None:
    #             mu_vec = grad_item.clone()
    #         else:
    #             mu_vec = torch.cat((mu_vec, grad_item.clone()), -1)
    #     mu_vec = mu_vec.cpu().numpy()  # (d)
    #     # print("5: ", mu_vec, mu_vec.shape)
    #     F_hat = np.outer(mu_vec, mu_vec)
    #     H_hat = A_hat * mu_vec
    #
    #     ret_val = - 1.0 / (1.0-self.opts.gamma) * H_hat
    #
    #     # print("6: ", F_hat.shape, H_hat.shape)
    #
    #     if w is None:
    #         return ret_val
    #     # print("7: ", (ret_val + np.dot(F_hat, w)).shape, np.dot(F_hat, w).shape)
    #     return ret_val + np.dot(F_hat, w)
    #
    #
    # def ANPG(self):
    #     x, v = None, None
    #     x_list = []
    #     for h in range(self.opts.H_npg):
    #         if h > 0:
    #             y = self.opts.alpha_npg * x + (1 - self.opts.alpha_npg) * v
    #             nabla_L = self._ANPG_sampling(w=y)
    #         else:
    #             nabla_L = self._ANPG_sampling(w=None)
    #             y = np.zeros_like(nabla_L)
    #             v = np.zeros_like(nabla_L)
    #         # print("1: ", nabla_L, nabla_L.shape)
    #         x = y - self.opts.delta_npg * nabla_L
    #         z = self.opts.beta_npg * y + (1 - self.opts.beta_npg) * v
    #         v = z - self.opts.xi_npg * nabla_L
    #
    #         if (h+1) > (self.opts.H_npg//2):
    #             x_list.append(x.copy())
    #
    #     grad = -1.0 * np.mean(x_list, axis=0) # (d, ) TODO: danger
    #     # print("2: ", grad, grad.shape)
    #     grad_list = []
    #     s_id = 0
    #     for item in self.parameters():
    #         e_id = s_id + np.prod(item.grad.shape)
    #         grad_item = torch.tensor(grad[s_id:e_id], dtype=item.grad.dtype, device=item.device).view(item.grad.shape)
    #         s_id = e_id
    #
    #         if self.is_Byzantine and self.attack_type is not None:
    #             if self.attack_type == 'zero-gradient':
    #                 grad_list.append(grad_item * 0)
    #             elif self.attack_type == 'random-noise':
    #                 rnd = (torch.rand(grad_item.shape, device=grad_item.device) * 2 - 1) * (
    #                             grad_item.max().data - grad_item.min().data) * 3
    #                 grad_list.append(grad_item + rnd)
    #             elif self.attack_type == 'sign-flipping':
    #                 grad_list.append(-2.5 * grad_item)
    #             elif self.attack_type == 'reward-flipping':
    #                 grad_list.append(grad_item)
    #             elif self.attack_type == 'random-action':
    #                 grad_list.append(grad_item)
    #             else:
    #                 raise NotImplementedError()
    #         else:
    #             grad_list.append(grad_item)
    #
    #     # print("3: ", grad_list, [g.shape for g in grad_list])
    #     # raise NotImplementedError
    #
    #     return grad_list, 0, 0, 0


    def to(self, device):
        self.logits_net.to(device)
        return self
    
    def eval(self):
        self.logits_net.eval()
        return self
        
    def train(self):
        self.logits_net.train()
        return self
    
    def parameters(self):
        return self.logits_net.parameters()
