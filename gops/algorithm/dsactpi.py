#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Distributed Soft Actor-Critic (DSAC) algorithm
#  Reference: Duan J, Guan Y, Li S E, et al.
#             Distributional soft actor-critic: Off-policy reinforcement learning
#             for addressing value estimation errors[J].
#             IEEE transactions on neural networks and learning systems, 2021.
#  Update: 2021-03-05, Ziqing Gu: create DSAC algorithm
#  Update: 2021-03-05, Wenxuan Wang: debug DSAC algorithm

__all__=["ApproxContainer","DSACTPI"]
import time
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.tensorboard_setup import tb_tags
from gops.utils.gops_typing import DataDict
from gops.utils.common_utils import get_apprfunc_dict


class ApproxContainer(ApprBase):
    """Approximate function container for DSAC.

    Contains one policy and one action value.
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)
        print(kwargs["device"])
        
        # create q networks
        q_args = get_apprfunc_dict("value", **kwargs)
        self.q1: nn.Module = create_apprfunc(**q_args).to(self.device_str)
        self.q2: nn.Module = create_apprfunc(**q_args).to(self.device_str)
        self.pi_net = self.q1.pi_net.to(self.device_str)
        self.q2.pi_net = self.pi_net.to(self.device_str)

        
        self.q1_target = deepcopy(self.q1).to(self.device_str)
        self.q2_target = deepcopy(self.q2).to(self.device_str)
        if kwargs["target_PI"]:
            self.pi_net_target = deepcopy(self.pi_net).to(self.device_str)   # use target pi
            self.q1_target.pi_net = self.pi_net_target.to(self.device_str)
            self.q2_target.pi_net = self.pi_net_target.to(self.device_str)
        else:
            self.q1_target.pi_net = self.pi_net.to(self.device_str)   # use online pi
            self.q2_target.pi_net = self.pi_net.to(self.device_str)

        # create policy network
        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args).to(self.device_str)
        self.policy.pi_net = self.pi_net.to(self.device_str)
        self.policy_target = deepcopy(self.policy).to(self.device_str)
        self.policy_target.pi_net = self.pi_net.to(self.device_str)

        # set target network gradients
        for p in self.policy_target.ego_paras():
            p.requires_grad = False
        for p in self.q1_target.ego_paras():
            p.requires_grad = False
        for p in self.q2_target.ego_paras():
            p.requires_grad = False
        if kwargs["target_PI"]:
            for p in self.pi_net_target.parameters():
                p.requires_grad = False

        # create entropy coefficient
        self.log_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32, device=self.device_str))

        # create optimizers
        self.q1_optimizer = Adam(self.q1.ego_paras(), lr=kwargs["value_learning_rate"])
        self.q2_optimizer = Adam(self.q2.ego_paras(), lr=kwargs["value_learning_rate"])
        self.policy_optimizer = Adam(
            self.policy.ego_paras(), lr=kwargs["policy_learning_rate"]
        )
        self.pi_optimizer = Adam(self.pi_net.parameters(), lr=kwargs["pi_learning_rate"])
        self.alpha_optimizer = Adam([self.log_alpha], lr=kwargs["alpha_learning_rate"])

        self.optimizer_dict = {
            "policy": self.policy_optimizer,
            "q1": self.q1_optimizer,
            "q2": self.q2_optimizer,
            "pi": self.pi_optimizer,
            "alpha": self.alpha_optimizer,
        }
        self.init_scheduler(**kwargs)


    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class DSACTPI(AlgorithmBase):
    """DSAC algorithm with three refinements, higher performance and more stable.

    Paper: https://arxiv.org/abs/2310.05858

    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    :param bool auto_alpha: whether to adjust temperature automatically.
    :param float alpha: initial temperature.
    :param float delay_update: delay update steps for actor.
    :param Optional[float] target_entropy: target entropy for automatic
        temperature adjustment.
    """

    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.gamma = kwargs["gamma"]
        self.tau = kwargs["tau"]
        self.target_entropy = -kwargs["action_dim"]
        self.auto_alpha = kwargs["auto_alpha"]
        self.alpha = kwargs.get("alpha", 0.2)
        self.delay_update = kwargs["delay_update"]
        self.mean_std1= None
        self.mean_std2= None
        self.tau_b = kwargs.get("tau_b", self.tau)
        self.target_PI = kwargs["target_PI"]
        self.per_flag = kwargs["buffer_name"].startswith("prioritized") # FIXME: hard code

    @property
    def adjustable_parameters(self):
        return (
            "gamma",
            "tau",
            "auto_alpha",
            "alpha",
            "delay_update",
        )

    def _local_update(self, data: DataDict, iteration: int) -> dict:
        tb_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return tb_info

    def get_remote_update_info(
        self, data: DataDict, iteration: int
    ) -> Tuple[dict, dict]:
        tb_info = self.__compute_gradient(data, iteration)

        update_info = {
            "q1_grad": [p._grad for p in self.networks.q1.ego_paras()],
            "q2_grad": [p._grad for p in self.networks.q2.ego_paras()],
            "policy_grad": [p._grad for p in self.networks.policy.ego_paras()],
             "pi_grad": [p._grad for p in self.networks.pi_net.parameters()],
            "iteration": iteration,
        }
        if self.auto_alpha:
            update_info.update({"log_alpha_grad":self.networks.log_alpha.grad})

        return tb_info, update_info

    def _remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        q1_grad = update_info["q1_grad"]
        q2_grad = update_info["q2_grad"]
        policy_grad = update_info["policy_grad"]
        pi_grad = update_info["pi_grad"]

        for p, grad in zip(self.networks.q1.ego_paras(), q1_grad):
            p._grad = grad
        for p, grad in zip(self.networks.q2.ego_paras(), q2_grad):
            p._grad = grad
        for p, grad in zip(self.networks.policy.ego_paras(), policy_grad):
            p._grad = grad
        for p, grad in zip(self.networks.pi_net.parameters(), pi_grad):
            p._grad = grad
        if self.auto_alpha:
            self.networks.log_alpha._grad = update_info["log_alpha_grad"]

        self.__update(iteration)

    def __get_alpha(self, requires_grad: bool = False):
        if self.auto_alpha:
            alpha = self.networks.log_alpha.exp()
            if requires_grad:
                return alpha
            else:
                return alpha.item()
        else:
            return self.alpha

    def __compute_gradient(self, data: DataDict, iteration: int):
        start_time = time.time()

        obs = data["obs"]
        logits = self.networks.policy(obs)
        logits_mean, logits_std = torch.chunk(logits, chunks=2, dim=-1)
        policy_mean = torch.tanh(logits_mean).mean().item()
        policy_std = logits_std.mean().item()

        act_dist = self.networks.create_action_distributions(logits)
        new_act, new_log_prob = act_dist.rsample()
        data.update({"new_act": new_act, "new_log_prob": new_log_prob})

        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        self.networks.policy_optimizer.zero_grad()
        self.networks.pi_optimizer.zero_grad()
        loss_q, q1, q2, std1, std2, origin_q_loss, idx, td_err  = self.__compute_loss_q(data)
        loss_q.backward()

        for p in self.networks.q1.ego_paras():
            p.requires_grad = False

        for p in self.networks.q2.ego_paras():
            p.requires_grad = False

 
        loss_policy, entropy = self.__compute_loss_policy(data)
        loss_policy.backward()

        for p in self.networks.q1.ego_paras():
            p.requires_grad = True
        for p in self.networks.q2.ego_paras():
            p.requires_grad = True

        if self.auto_alpha:
            self.networks.alpha_optimizer.zero_grad()
            loss_alpha = self.__compute_loss_alpha(data)
            loss_alpha.backward()

        # calculate gradient norm
        q1_grad_norm = torch.norm( torch.cat([p.grad.flatten() for p in self.networks.q1.ego_paras()]))
        q2_grad_norm = torch.norm( torch.cat([p.grad.flatten() for p in self.networks.q2.ego_paras()]))
        policy_grad_norm = torch.norm( torch.cat([p.grad.flatten() for p in self.networks.policy.ego_paras()]))
        pi_grad_norm = torch.norm( torch.cat([p.grad.flatten() for p in self.networks.pi_net.parameters()]))
        tb_info = {
            "DSAC2/critic_avg_q1-RL iter": q1.mean().detach().item(),
            "DSAC2/critic_avg_q2-RL iter": q2.mean().detach().item(),
            "DSAC2/critic_avg_std1-RL iter": std1.mean().detach().item(),
            "DSAC2/critic_avg_std2-RL iter": std2.mean().detach().item(),
            "DSAC2/critic_avg_min_std1-RL iter": std1.min().detach().item(),
            "DSAC2/critic_avg_min_std2-RL iter": std2.min().detach().item(),
            "DSAC2/critic_avg_max_std1-RL iter": std1.max().detach().item(),
            "DSAC2/critic_avg_max_std2-RL iter": std2.max().detach().item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            tb_tags["loss_critic"]: origin_q_loss.item(),
            "DSAC2/policy_mean-RL iter": policy_mean,
            "DSAC2/policy_std-RL iter": policy_std,
            "DSAC2/entropy-RL iter": entropy.item(),
            "DSAC2/alpha-RL iter": self.__get_alpha(),
            "DSAC2/mean_std1": self.mean_std1,
            "DSAC2/mean_std2": self.mean_std2,
            "DSAC2/q_grad_norm": (q1_grad_norm+ q2_grad_norm).item()/2,
            "DSAC2/policy_grad_norm": policy_grad_norm.item(),
            "DSAC2/pi_grad_norm": pi_grad_norm.item(),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }
        if self.per_flag:
            return tb_info, idx, td_err
        else:
            return tb_info

    def __q_evaluate(self, obs, act, qnet):
        StochaQ = qnet(obs, act)
        mean, std = StochaQ[..., 0], StochaQ[..., -1]
        normal = Normal(torch.zeros_like(mean), torch.ones_like(std))
        z = normal.sample()
        z = torch.clamp(z, -3, 3)
        q_value = mean + torch.mul(z, std)
        return mean, std, q_value

    def __compute_loss_q(self, data: DataDict):
        obs, act, rew, obs2, done = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        if self.per_flag:
            weight = data["weight"]
        else:
            weight = 1.0
        with torch.no_grad():
            logits_2 = self.networks.policy_target(obs2)
            act2_dist = self.networks.create_action_distributions(logits_2)
            act2, log_prob_act2 = act2_dist.rsample()

        q1, q1_std, _ = self.__q_evaluate(obs, act, self.networks.q1)

        q2, q2_std, _ = self.__q_evaluate(obs, act, self.networks.q2)
        if self.mean_std1 is None:
            self.mean_std1 = torch.mean(q1_std.detach())
        else:
            self.mean_std1 = (1 - self.tau_b) * self.mean_std1 + self.tau_b * torch.mean(q1_std.detach())

        if self.mean_std2 is None:
            self.mean_std2 = torch.mean(q2_std.detach())
        else:
            self.mean_std2 = (1 - self.tau_b) * self.mean_std2 + self.tau_b * torch.mean(q2_std.detach())

        with torch.no_grad():
            q1_next, _, q1_next_sample = self.__q_evaluate(
                obs2, act2, self.networks.q1_target
            )
            
            q2_next, _, q2_next_sample = self.__q_evaluate(
                obs2, act2, self.networks.q2_target
            )
            q_next = torch.min(q1_next, q2_next)
            q_next_sample = torch.where(q1_next < q2_next, q1_next_sample, q2_next_sample)

        target_q1, target_q1_bound = self.__compute_target_q(
            rew,
            done,
            q1.detach(),
            self.mean_std1.detach(),
            q_next.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )
        
        target_q2, target_q2_bound = self.__compute_target_q(
            rew,
            done,
            q2.detach(),
            self.mean_std2.detach(),
            q_next.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )

        q1_std_detach = torch.clamp(q1_std, min=0.).detach()
        q2_std_detach = torch.clamp(q2_std, min=0.).detach()
        bias = 0.1

        q1_loss = (torch.pow(self.mean_std1, 2) + bias) * torch.mean(weight*(
            -(target_q1 - q1).detach() / ( torch.pow(q1_std_detach, 2)+ bias)*q1
            -((torch.pow(q1.detach() - target_q1_bound, 2)- q1_std_detach.pow(2) )/ (torch.pow(q1_std_detach, 3) +bias)
            )*q1_std)
        )

        q2_loss = (torch.pow(self.mean_std2, 2) + bias)*torch.mean(weight*(
            -(target_q2 - q2).detach() / ( torch.pow(q2_std_detach, 2)+ bias)*q2
            -((torch.pow(q2.detach() - target_q2_bound, 2)- q2_std_detach.pow(2) )/ (torch.pow(q2_std_detach, 3) +bias)
            )*q2_std)
        )
        with torch.no_grad():
            origin_q1_loss = (torch.pow(self.mean_std1, 2)) * torch.mean(
                torch.pow((target_q1 - q1),2) / ( torch.pow(q1_std_detach, 2)+ 1e-6)  
                + torch.log(q1_std_detach+1e-6)) # for numerical stability
            origin_q2_loss = (torch.pow(self.mean_std2, 2)) * torch.mean(
                torch.pow((target_q2 - q2),2) / ( torch.pow(q2_std_detach, 2)+ 1e-6)  
                + torch.log(q2_std_detach+1e-6))
            origin_q_loss = origin_q1_loss + origin_q2_loss
        

        if self.per_flag:
            idx = data["idx"]
            td_err = (torch.abs(target_q1 - q1) + torch.abs(target_q2 - q2)) / 2
            # print("td_err_max", td_err.max().item())
            # print("td_err_min", td_err.min().item())
            per = td_err/2000 # TODO: 2000 is a hyperparameter
        else:
            idx = None
            per = None

        return q1_loss + q2_loss, q1, q2, q1_std, q2_std, origin_q_loss, idx, per

    def __compute_target_q(self, r, done, q,q_std, q_next, q_next_sample, log_prob_a_next):
        target_q = r + (1 - done) * self.gamma * (
            q_next - self.__get_alpha() * log_prob_a_next
        )
        target_q_sample = r + (1 - done) * self.gamma * (
            q_next_sample - self.__get_alpha() * log_prob_a_next
        )
        td_bound = 3 * q_std
        difference = torch.clamp(target_q_sample - q, -td_bound, td_bound)
        target_q_bound = q + difference
        return target_q.detach(), target_q_bound.detach()

    def __compute_loss_policy(self, data: DataDict):
        obs, new_act, new_log_prob = data["obs"], data["new_act"], data["new_log_prob"]
        q1, _, _ = self.__q_evaluate(obs, new_act, self.networks.q1)
        q2, _, _ = self.__q_evaluate(obs, new_act, self.networks.q2)
        loss_policy = (self.__get_alpha() * new_log_prob - torch.min(q1,q2)).mean()
        entropy = -new_log_prob.detach().mean()
        return loss_policy, entropy

    def __compute_loss_alpha(self, data: DataDict):
        new_log_prob = data["new_log_prob"]
        loss_alpha = (
            -self.networks.log_alpha
            * (new_log_prob.detach() + self.target_entropy).mean()
        )
        return loss_alpha

    def __update(self, iteration: int):
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()
        self.networks.pi_optimizer.step()

        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()

            if self.auto_alpha:
                self.networks.alpha_optimizer.step()

            with torch.no_grad():
                polyak = 1 - self.tau
                for p, p_targ in zip(
                    self.networks.q1.ego_paras(), self.networks.q1_target.ego_paras()
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                for p, p_targ in zip(
                    self.networks.q2.ego_paras(), self.networks.q2_target.ego_paras()
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                for p, p_targ in zip(
                    self.networks.policy.ego_paras(),
                    self.networks.policy_target.ego_paras(),
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

                if self.target_PI:
                    for p, p_targ in zip(
                        self.networks.pi_net.parameters(),
                        self.networks.pi_net_target.parameters(),
                    ):
                        p_targ.data.mul_(polyak)
                        p_targ.data.add_((1 - polyak) * p.data)