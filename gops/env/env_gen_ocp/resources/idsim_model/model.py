from typing import Tuple
from copy import deepcopy

import torch

# from idsim.config import Config
from gops.env.env_gen_ocp.resources.idsim_var_type import Config

from gops.env.env_gen_ocp.resources.idsim_model.observe.ego import get_ego_obs
from gops.env.env_gen_ocp.resources.idsim_model.observe.ref import select_ref_by_index, compute_onref_mask, get_ref_obs, get_ref_obs_frenet_coord
from gops.env.env_gen_ocp.resources.idsim_model.observe.sur import get_sur_obs

from gops.env.env_gen_ocp.resources.idsim_model.utils.math_utils import convert_ground_coord_to_ego_coord, convert_ref_to_ego_coord, cal_curvature
from gops.env.env_gen_ocp.resources.idsim_model.utils.model_utils import stack_samples_full_horizon, select_sample
from gops.env.env_gen_ocp.resources.idsim_model.utils.reward_utils import square_loss, dist_3to2_circles

from gops.env.env_gen_ocp.resources.idsim_model.model_context import State
from gops.env.env_gen_ocp.resources.idsim_model.model_context import BaseContext
from gops.env.env_gen_ocp.resources.idsim_model.params import ModelConfig


class IdSimModel:
    def __init__(self, env_config: Config, model_config: ModelConfig):
        # action bounds
        self.action_lower_bound = torch.tensor(env_config.action_lower_bound)
        self.action_upper_bound = torch.tensor(env_config.action_upper_bound)
        self.action_center = (self.action_upper_bound + self.action_lower_bound) / 2
        self.action_half_range = (self.action_upper_bound - self.action_lower_bound) / 2
        self.real_action_upper = torch.tensor(
            model_config.real_action_upper, dtype=torch.float32)
        self.real_action_lower = torch.tensor(
            model_config.real_action_lower, dtype=torch.float32)

        self.Ts = env_config.dt
        self.vehicle_spec = env_config.vehicle_spec
        self.ref_v = model_config.ref_v_lane
        self.N = model_config.N
        obs_num_surr_dict = env_config.obs_num_surrounding_vehicles
        self.M = obs_num_surr_dict['passenger'] + \
                 obs_num_surr_dict['bicycle'] + obs_num_surr_dict['pedestrian']
        self.model_config: ModelConfig = model_config
        self.obs_dim = self.get_obs_dim()

    def get_obs_dim(self):
        obs_dim = self.model_config.ego_feat_dim + \
            (self.model_config.per_sur_feat_dim + 3) * self.M + \
            self.model_config.per_ref_feat_dim * len(self.model_config.downsample_ref_point_index)
        if self.model_config.add_boundary_obs:
            obs_dim += self.model_config.ego_bound_dim
        return obs_dim

    def observe(self, context: BaseContext) -> torch.Tensor:
        ego_obs = get_ego_obs(context)
        if self.model_config.full_horizon_sur_obs:
            sur_obs = get_sur_obs(context, self.N + 1)
        else:
            sur_obs = get_sur_obs(context, 1)
        # multi_ref_obs = get_ref_obs(context, self.model_config.num_ref_points, self.model_config.num_ref_lines)
        multi_ref_obs = get_ref_obs_frenet_coord(context, self.model_config.num_ref_points, self.model_config.num_ref_lines)
        ref_index = context.p.ref_index_param
        ref_obs = select_ref_by_index(multi_ref_obs, ref_index)
        downsample_ref_point_index = torch.tensor(self.model_config.downsample_ref_point_index)
        downsample_index = torch.stack(
            [downsample_ref_point_index + i * self.model_config.num_ref_points
            for i in range(self.model_config.per_ref_feat_dim)]
        ).flatten()
        ref_obs = ref_obs[:, downsample_index]
        # boundary
        if self.model_config.add_boundary_obs:
            bound_obs = context.p.boundary_param
            obs = torch.cat((ego_obs, ref_obs, sur_obs, bound_obs), dim=-1)
        else:
            obs = torch.cat((ego_obs, ref_obs, sur_obs), dim=-1)
        return obs

    def dynamics(self, context: BaseContext, action: torch.Tensor) -> State:
        ego_state = context.x.ego_state
        last_action = context.x.last_action  # real value
        action = self.inverse_normalize_action(action)
        action = last_action + action  # real value
        action = self.action_clamp(action)
        next_ego_state = ego_predict_model(ego_state, action, self.Ts, self.vehicle_spec)
        return State(ego_state=next_ego_state, last_last_action=last_action, last_action=action)

    def dynamics_full_horizon(self,
                              context: BaseContext,
                              action: torch.Tensor) -> Tuple[BaseContext, torch.Tensor, torch.Tensor]:
        # practically implemented action in real action range
        init_last_last_action = context.x.last_last_action
        # practically implemented action in real action range
        init_last_action = context.x.last_action
        context_list = []
        for step in range(self.N):
            # normalized action outputted by policy
            current_action = action[:, step * 2: step * 2 + 2]
            next_state = self.dynamics(context, current_action)
            context = context.advance(next_state)
            context_list.append(context)
        context_full = stack_samples_full_horizon(context_list)
        # [B, N, 2]
        last_last_action_full = context_full.x.last_last_action
        last_action_full = context_full.x.last_action
        # shift backward one step
        # when context(t+1), we have last_last_action(t-2), last_action(t-1), action(t)
        # in context: last_last_action(t-1), last_action(t)
        # action \in [-1, 1], the policy output
        # last_last_action and last_action \in [real_action_lower, real_action_upper], the real action
        last_last_action_full = torch.cat(
            (init_last_last_action.unsqueeze(1), last_last_action_full[:, :-1]), dim=1)  # [B, N, 2]
        last_action_full = torch.cat(
            (init_last_action.unsqueeze(1), last_action_full[:, :-1]), dim=1)  # [B, N, 2]
        return context_full, last_last_action_full, last_action_full

    def reward_nn_state(self, context: BaseContext,  # context of next state
                        last_last_action: torch.Tensor,  # last_last_action of state
                        last_action: torch.Tensor,  # last_action of state
                        action: torch.Tensor  # (normalized) action of state
                        ):
        ego_state = context.x.ego_state  # [B, 6] # x, y, vx, vy, phi, r
        ego_x, ego_y, ego_vx, ego_vy, ego_phi, ego_r = ego_state.unbind(dim=-1) # [B,]
        B = ego_state.shape[0]  # scalar
        ref_param = context.p.ref_param  # [B, R, 2N+1, 4] # x, y, phi, speed
        # [B, 2*N+1, M, 7] # x, y, phi, speed, length, width, mask
        sur_param = context.p.sur_param
        action_real = context.x.last_action  # [B, 2] , action_real == last_action of next_state
        ref_index_param = context.p.ref_index_param  # [B,]
        sur_state = sur_param[:, context.i]  # [B, M, 7]
        onref_mask = compute_onref_mask(context)  # [B, M]
        with torch.no_grad():
            if self.model_config.track_closest_ref_point:
                ref_line = ref_param[torch.arange(B), ref_index_param, 0:self.model_config.num_ref_points, :]  # [B, N+1, 4]
                ref_line_x, ref_line_y = ref_line[:, :, :2].unbind(dim=-1)  # [B, N+1]
                ref_line_distance = torch.square(ref_line_x - ego_x.unsqueeze(-1)) + torch.square(ref_line_y - ego_y.unsqueeze(-1)) # [B, N+1]
                ref_state_index = torch.argmin(ref_line_distance, dim=-1)  # [B,]
            else:
                ref_state_index = context.i * torch.ones_like(ego_x, dtype=torch.int)   # [B,]
        ref_state = ref_param[torch.arange(B), ref_index_param, ref_state_index, :]  # [B, 4]

        # nominal action
        if self.model_config.use_nominal_action:
            nominal_acc = self._get_nominal_acc_by_state(ref_param, ref_index_param, context.p.light_param, ref_state_index)
            nominal_steer = self._get_nominal_steer_by_state(ego_state, ref_param, ref_index_param, ref_state_index)
        else:
            nominal_acc = 0
            nominal_steer = 0
        return self.get_reward_by_state(ego_state, ref_state, sur_state,
                                        last_last_action, last_action, action, action_real,
                                        nominal_acc, nominal_steer, onref_mask, context)

    def reward_full_horizon(self, context_full: BaseContext,
                            last_last_action_full: torch.Tensor,
                            last_action_full: torch.Tensor,
                            action_full: torch.Tensor) -> torch.Tensor:
        ego_state = context_full.x.ego_state # [B, N, 6]
        B = ego_state.shape[0]
        i = context_full.i  # [N, ]
        # add batch
        i = i.unsqueeze(0).repeat(B, 1)  # [B, N]
        # [B, N, R, 2N+1, 4] # x, y, phi, speed
        ref_param = context_full.p.ref_param
        ref_index_param = context_full.p.ref_index_param  # [B, N]
        # [B, N, 2N+1, M, 7] # x, y, phi, speed, length, width, mask
        sur_param = context_full.p.sur_param
        light_param = context_full.p.light_param
        N = self.N
        R = ref_param.shape[2]
        M = sur_param.shape[3]
        # flatten the first 2 dimensions
        ego_state_full = ego_state.reshape(-1, 6)  # [B*N, 6]
        ego_x_full, ego_y_full, _, _, _, _ = ego_state_full.unbind(dim=-1) # [B*N,]
        # [B*N, R, 2N+1, 4]
        ref_param_full = ref_param.reshape(-1, R, 2 * self.N + 1, 4)
        # [B*N, 2N+1, M, 7]
        sur_param_full = sur_param.reshape(-1, 2 * self.N + 1, self.M, 7)
        # [B*N, N+1, 3]
        light_param_full = light_param.reshape(-1, self.N + 1, 3)
        ref_index_param_full = ref_index_param.reshape(-1)  # [B*N, ]

        i_full = i.reshape(-1)  # [B*N, ]
        action_real = context_full.x.last_action  # [B, N, 2]
        action_real_full = action_real.reshape(-1, 2)  # [B*N, 2]
        # reduce R dim: select ref path [B*N, R, 2N+1, 4] -> [B*N, 2N+1, 4]
        ref_state_full = ref_param_full[torch.arange(
            B * N), ref_index_param_full, :, :]

        # reduce 2N+1 dim: select cur sur state [B, 2N+1, M, 7] -> [B, M, 7]
        sur_state_full = sur_param_full[torch.arange(
            B * N), i_full, :, :]

        with torch.no_grad():
            # ref state index
            if self.model_config.track_closest_ref_point:
                ref_line = ref_param_full[torch.arange(B*N), ref_index_param_full, 0:self.model_config.num_ref_points, :]  # [B*N, R, 2N+1, 4]
                ref_line_x, ref_line_y = ref_line[:, :, :2].unbind(dim=-1)  # [B*N, N+1]
                ref_line_distance = torch.square(ref_line_x - ego_x_full.unsqueeze(-1)) + torch.square(ref_line_y - ego_y_full.unsqueeze(-1)) # [B*N, N+1]
                ref_state_index_full = torch.argmin(ref_line_distance, dim=-1)  # [B*N,]
            else:
                ref_state_index_full = i_full   # [B*N,]
        # reduce 2N+1 dim: select cur ref point [B*N, 2N+1, 4] -> [B*N, 4]
        ref_state_full = ref_state_full[torch.arange(
            B * N), ref_state_index_full, :]

        context_first = select_sample(context_full, 0)
        onref_mask = compute_onref_mask(
            context_first)  # [B, M]
        # repeat N times -> [B, N, M]
        onref_mask = onref_mask.unsqueeze(1).repeat(1, N, 1)
        onref_mask = onref_mask.reshape(-1, M)  # [B*N, M]

        last_last_action_full = last_last_action_full.reshape(-1, 2)
        last_action_full = last_action_full.reshape(-1, 2)
        action_full = action_full.reshape(-1, N, 2)
        action_full = action_full.reshape(-1, 2)

        # nominal action
        if self.model_config.use_nominal_action:
            nominal_acc_full = self._get_nominal_acc_by_state(ref_param_full, ref_index_param_full, light_param_full, ref_state_index_full)
            nominal_steer_full = self._get_nominal_steer_by_state(ego_state_full, ref_param_full, ref_index_param_full, ref_state_index_full)
        else:
            nominal_acc_full = 0
            nominal_steer_full = 0
        return self.get_reward_by_state(ego_state_full, ref_state_full, sur_state_full,
                                        last_last_action_full, last_action_full,
                                        action_full, action_real_full,
                                        nominal_acc_full, nominal_steer_full,
                                        onref_mask, context_full)

    def get_reward_by_state(self, ego_state: torch.Tensor,
                            ref_state: torch.Tensor,
                            sur_state: torch.Tensor,
                            last_last_action: torch.Tensor,  # a_0|t-2 if i==0, a_0|t-1 if i==1, a_i-2|t if i>=2
                            last_action: torch.Tensor,  # a_0|t-1 if i==0, a_i-1|t if i>=1
                            action: torch.Tensor,
                            action_real: torch.Tensor,  # a_i|t (with grad)
                            nominal_acc: torch.Tensor,
                            nominal_steer: torch.Tensor,
                            onref_mask: torch.Tensor,
                            context: BaseContext = None
                            ):
        model_config = deepcopy(self.model_config)
        # ego
        ego_x, ego_y, ego_vx, ego_vy, ego_phi, ego_r = ego_state.unbind(dim=-1) # [B,]
        # ref
        ref_x, ref_y, ref_phi = ref_state[:, :3].unbind(dim=-1)
        # ref_x_ego_coord, ref_y_ego_coord, ref_phi_ego_coord = \
        #     convert_ground_coord_to_ego_coord(ref_x, ref_y, ref_phi,
        #                                       ego_x, ego_y, ego_phi)  # [B,], ego coordinate
        ref_x_ego_coord, ref_y_ego_coord, ref_phi_ego_coord = \
            convert_ground_coord_to_ego_coord(ego_x, ego_y, ego_phi,
                                              ref_x, ref_y, ref_phi)  # [B,], frenet coordinate

        # tracking cost (multiplied by Q)
        cost_tracking_lon = square_loss(ref_x_ego_coord)
        cost_tracking_lat = square_loss(ref_y_ego_coord)
        cost_tracking_vx = square_loss(ref_state[:, -1] - ego_vx)
        cost_tracking_vy = square_loss(ego_state[:, 3])
        cost_tracking_phi = square_loss(ref_phi_ego_coord)
        cost_tracking_yaw_rate = square_loss(ego_state[:, 5])

        # soft action penalty
        ## 0-th order constraint (guaranteed by model.action_clamp)
        ## 1-th order constraint (guaranteed by tanh layer of network) (but still need penalty)
        acc_rate_1 = (action_real - last_action)[:, 0] / self.Ts
        steer_rate_1 = (action_real - last_action)[:, 1] / self.Ts
        cost_acc_rate_1 = square_loss(acc_rate_1)
        cost_steer_rate_1 = square_loss(steer_rate_1)
        ## 2-th order constraint (only steer is required by IDC) (multiplied by C_steer_rate_2)
        steer_rate_2 = (action_real - 2 * last_action + last_last_action)[:, 1] / self.Ts / self.Ts
        cost_steer_rate_2_min = (steer_rate_2 < model_config.steer_rate_2_min) * square_loss(
            steer_rate_2 - model_config.steer_rate_2_min)
        cost_steer_rate_2_max = (steer_rate_2 > model_config.steer_rate_2_max) * square_loss(
            steer_rate_2 - model_config.steer_rate_2_max)

        # soft 0-th order state penalty (multiplied by C_v)
        cost_vx_min = (ego_vx < model_config.vx_min) * square_loss(ego_vx - model_config.vx_min)
        cost_vx_max = (ego_vx > model_config.vx_max) * square_loss(ego_vx - model_config.vx_max)
        cost_vy_min = (ego_vy < model_config.vy_min) * square_loss(ego_vy - model_config.vy_min)
        cost_vy_max = (ego_vy > model_config.vy_max) * square_loss(ego_vy - model_config.vy_max)

        # penalty terms
        ## lateral error penalty
        # lat_err_from_ego_to_ref_line = - (ego_x - ref_x) * torch.sin(ref_phi) + (ego_y - ref_y) * torch.cos(ref_phi)
        # cost_penalty_lat_error = square_loss(lat_err_from_ego_to_ref_line)
        cost_penalty_lat_error = torch.zeros_like(ego_x)

        ## safe cost penalty
        sur_x, sur_y, sur_phi, sur_vx, sur_length, sur_width, sur_mask = sur_state.unbind(dim=-1)  # [B, M]
        rel_x_ego_coord, rel_y_ego_coord, rel_phi_ego_coord = convert_ground_coord_to_ego_coord(
            sur_x, sur_y, sur_phi, ego_x.unsqueeze(dim=-1), ego_y.unsqueeze(dim=-1), ego_phi.unsqueeze(dim=-1)
        )
        circle_dists, safe_dist = dist_3to2_circles(
            rel_x_ego_coord + 1e-5,
            rel_y_ego_coord + 1e-5,
            torch.cos(rel_phi_ego_coord),
            torch.sin(rel_phi_ego_coord),
            sur_length,  # length
            sur_width,  # width
            model_config.ego_length,
            model_config.ego_width,
        )  # [Batch(B), num_obs(M), ego_circle_num(3), sur_circle_num(2)], [Batch(B), num_obs(M), 1, 1]
        diff_dist = circle_dists - safe_dist - model_config.safe_dist_incremental
        collision_flag = ((circle_dists <= safe_dist).sum(dim=(-2, -1)) > 0) * sur_mask  # [B, M]
        is_sur_behind_ego = rel_x_ego_coord < 0
        C_back = model_config.C_back[0] * is_sur_behind_ego + model_config.C_back[1] * (~is_sur_behind_ego)
        cost_safe = torch.square(
            (diff_dist < 0) * diff_dist).sum(dim=(-2, -1)) * C_back * sur_mask * onref_mask  # [B, M]

        if model_config.clear_nonessential_cost_safe:
            cost_safe_reset_mask_1 = (ego_vx < 0.01).unsqueeze(-1).detach() # [B, 1], trick：自车停止时，清空周车距离惩罚
            cost_safe_reset_mask_2 = ((torch.abs(rel_y_ego_coord) > (safe_dist.squeeze(-1).squeeze(-1) + 1.2)) 
                                    * (torch.abs(torch.sin(rel_phi_ego_coord)) < 0.0872)
                                    * sur_mask).detach().bool() # [B, M]，trick：清空相邻车道且朝向一致或相反障碍物的距离惩罚
            cost_safe = cost_safe * (~cost_safe_reset_mask_1) * (~cost_safe_reset_mask_2)
        
        # action cost (multiplied by R)
        collision_mask = collision_flag.sum(-1) # [B,]
        nominal_acc_update_mask = (
            ((collision_mask > 0) * (self.model_config.C_obs > 0))
            ) > 0
        nominal_acc = (~nominal_acc_update_mask) * nominal_acc + nominal_acc_update_mask * self.real_action_lower[0]
        cost_action_real_acc = square_loss(action_real[:, 0] - nominal_acc)
        cost_action_real_steer = square_loss(action_real[:, 1] - nominal_steer)

        # C road
        cost_road = torch.zeros_like(ego_x)  # [B, ]

        (
            reward_tracking_lon,
            reward_tracking_lat,
            reward_tracking_phi,
            reward_tracking_vx,
            reward_tracking_vy,
            reward_tracking_yaw_rate,

            reward_action_acc,
            reward_action_steer,

            reward_cost_acc_rate_1,
            reward_cost_steer_rate_1,
            reward_cost_steer_rate_2_min,
            reward_cost_steer_rate_2_max,

            reward_cost_vx_min,
            reward_cost_vx_max,
            reward_cost_vy_min,
            reward_cost_vy_max,

            reward_penalty_lat_error,
            reward_penalty_sur_dist,
            reward_penalty_road,

            collision_flag,
        ) = (
            -cost_tracking_lon * model_config.reward_scale,
            -cost_tracking_lat * model_config.reward_scale,
            -cost_tracking_phi * model_config.reward_scale,
            -cost_tracking_vx * model_config.reward_scale,
            -cost_tracking_vy * model_config.reward_scale,
            -cost_tracking_yaw_rate * model_config.reward_scale,

            -cost_action_real_acc * model_config.reward_scale,
            -cost_action_real_steer * model_config.reward_scale,

            -cost_acc_rate_1 * model_config.reward_scale,
            -cost_steer_rate_1 * model_config.reward_scale,
            -cost_steer_rate_2_min * model_config.reward_scale,
            -cost_steer_rate_2_max * model_config.reward_scale,

            -cost_vx_min * model_config.reward_scale,
            -cost_vx_max * model_config.reward_scale,
            -cost_vy_min * model_config.reward_scale,
            -cost_vy_max * model_config.reward_scale,

            -cost_penalty_lat_error * model_config.reward_scale,
            -cost_safe.sum(-1) * model_config.reward_scale,  # [B,]
            -cost_road * model_config.reward_scale,

            collision_flag.max(dim=1)[0],
        )

        # focus more on low speed tracking, since it relates to stopping at red light and steering around obstacles
        model_config.Q[0] = model_config.Q[0] + (torch.abs(ref_state[:, -1]) < model_config.ref_v_slow_focus).detach() * model_config.Q_slow_incre[0] #trick：加强参考低速纵向跟踪
        model_config.Q[1] = model_config.Q[1] + (torch.abs(ego_vx) < model_config.ref_v_slow_focus).detach() * model_config.Q_slow_incre[1] #trick：减弱自车低速横向跟踪，Q_slow_incre[1]应为负值，起步和停车时不要过大转向
        model_config.Q[2] = model_config.Q[2] + (torch.abs(ref_state[:, -1]) < model_config.ref_v_slow_focus).detach() * model_config.Q_slow_incre[2] #trick：加强参考低速跟踪，停车
        model_config.Q[3] = model_config.Q[3] + (torch.abs(ref_state[:, -1]) < model_config.ref_v_slow_focus).detach() * model_config.Q_slow_incre[3] #trick：加强参考低速跟踪，停车
        model_config.Q[4] = model_config.Q[4] + (torch.abs(ref_state[:, -1]) < model_config.ref_v_slow_focus).detach() * model_config.Q_slow_incre[4] #trick：加强参考低速跟踪，停车
        model_config.Q[5] = model_config.Q[5] + (torch.abs(ref_state[:, -1]) < model_config.ref_v_slow_focus).detach() * model_config.Q_slow_incre[5] #trick：加强参考低速跟踪，停车
        model_config.R[0] = model_config.R[0] + (torch.abs(ref_state[:, -1]) < model_config.ref_v_slow_focus).detach() * model_config.R_slow_incre[0] #trick：加强参考低速跟踪，停车
        model_config.R[1] = model_config.R[1] + (torch.abs(ref_state[:, -1]) < model_config.ref_v_slow_focus).detach() * model_config.R_slow_incre[1] #trick：限制参考低速前轮转角

        reward = (
            model_config.Q[0] * reward_tracking_lon
            + model_config.Q[1] * reward_tracking_lat
            + model_config.Q[2] * reward_tracking_phi
            + model_config.Q[3] * reward_tracking_vx
            + model_config.Q[4] * reward_tracking_vy
            + model_config.Q[5] * reward_tracking_yaw_rate

            + model_config.R[0] * reward_action_acc
            + model_config.R[1] * reward_action_steer

            + model_config.C_acc_rate_1 * reward_cost_acc_rate_1
            + model_config.C_steer_rate_1 * reward_cost_steer_rate_1
            + model_config.C_steer_rate_2[0] * reward_cost_steer_rate_2_min
            + model_config.C_steer_rate_2[1] * reward_cost_steer_rate_2_max

            + model_config.C_v[0] * reward_cost_vx_min
            + model_config.C_v[1] * reward_cost_vx_max
            + model_config.C_v[2] * reward_cost_vy_min
            + model_config.C_v[3] * reward_cost_vy_max

            + model_config.C_lat * reward_penalty_lat_error
            + model_config.C_obs * reward_penalty_sur_dist
            + model_config.C_road * reward_penalty_road
        )

        return (
            reward,
            model_config.Q[0] * reward_tracking_lon,
            model_config.Q[1] * reward_tracking_lat,
            model_config.Q[2] * reward_tracking_phi,
            model_config.Q[3] * reward_tracking_vx,
            model_config.Q[4] * reward_tracking_vy,
            model_config.Q[5] * reward_tracking_yaw_rate,

            model_config.R[0] * reward_action_acc,
            model_config.R[1] * reward_action_steer,

            model_config.C_acc_rate_1 * reward_cost_acc_rate_1,
            model_config.C_steer_rate_1 * reward_cost_steer_rate_1,
            model_config.C_steer_rate_2[0] * reward_cost_steer_rate_2_min,
            model_config.C_steer_rate_2[1] * reward_cost_steer_rate_2_max,

            model_config.C_v[0] * reward_cost_vx_min,
            model_config.C_v[1] * reward_cost_vx_max,
            model_config.C_v[2] * reward_cost_vy_min,
            model_config.C_v[3] * reward_cost_vy_max,

            model_config.C_lat * reward_penalty_lat_error,
            model_config.C_obs * reward_penalty_sur_dist,
            model_config.C_road * reward_penalty_road,

            collision_flag,
        )

    def _get_nominal_steer_by_state(self, ego_state: torch.Tensor,
                                    ref_param: torch.Tensor,
                                    ref_index: torch.Tensor,
                                    ref_state_index: torch.Tensor
                                    ) -> torch.Tensor:
        # ref_param: [B, R, 2N+1, 4]
        # use ref_index to select ref_param, remove R
        # use ref_state_index to determine the start, from 2N+1 to 3
        # ref_line: [B, 3, 4]

        ref_line = torch.stack([
            ref_param[torch.arange(ref_param.shape[0]),
            ref_index, ref_state_index+i, :] for i in [0, 5, 10]], dim=1)
        ref_x_ego_coord, ref_y_ego_coord, ref_phi_ego_coord = \
            convert_ref_to_ego_coord(
                ref_line[:, :, :3], ego_state)  # [B, N+1]
        # nominal action
        x1, y1 = ref_x_ego_coord[:, 0], ref_y_ego_coord[:, 0]  # [B,]
        x2, y2 = ref_x_ego_coord[:, 1], ref_y_ego_coord[:, 1]
        x3, y3 = ref_x_ego_coord[:, 2], ref_y_ego_coord[:, 2]
        nominal_curvature = cal_curvature(x1, y1, x2, y2, x3, y3)
        nominal_steer = nominal_curvature * 2.65 #FIXME: hard-coded: wheel base
        nominal_steer = torch.clamp(
            nominal_steer, self.real_action_lower[1], self.real_action_upper[1])
        return nominal_steer.detach()

    def _get_nominal_acc_by_state(self,
                                  ref_param: torch.Tensor,
                                  ref_index: torch.Tensor,
                                  light_param: torch.Tensor,
                                  ref_state_index: torch.Tensor,
                                  ) -> torch.Tensor:
        ref_vx_0 = ref_param[torch.arange(ref_param.shape[0]),
            ref_index, ref_state_index, 3]
        ref_vx_1 = ref_param[torch.arange(ref_param.shape[0]),
            ref_index, ref_state_index+1, 3]
        nominal_acc = (ref_vx_1 - ref_vx_0) / self.Ts

        light_param = light_param[torch.arange(ref_param.shape[0]),
            ref_state_index, :2]
        traffic_light, ahead_lane_length = light_param.unbind(dim=-1)
        red_mask = (traffic_light != 0) \
            * (ahead_lane_length <= self.model_config.ahead_lane_length_min) \
            * (ahead_lane_length != -1)
        nominal_acc_update_mask = (red_mask > 0)
        nominal_acc = (~nominal_acc_update_mask) * nominal_acc + nominal_acc_update_mask * self.real_action_lower[0]
        return nominal_acc.detach()

    def done(self, context: BaseContext) -> bool:
        raise NotImplementedError

    def action_clamp(self, action: torch.Tensor) -> torch.Tensor:
        self.real_action_upper = self.real_action_upper.to(self.device)
        self.real_action_lower = self.real_action_lower.to(self.device)
        # clip action to real action range
        action = action - (action > self.real_action_upper) * \
                 (action.detach() - self.real_action_upper)
        action = action - (action < self.real_action_lower) * \
                 (action.detach() - self.real_action_lower)
        return action

    def inverse_normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        self.device = action.device
        self.action_half_range = self.action_half_range.to(self.device)
        self.action_center = self.action_center.to(self.device)
        action = action * self.action_half_range + self.action_center
        return action

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        action = (action - self.action_center) / self.action_half_range
        return action


def ego_predict_model(ego_state: torch.Tensor,
                      action: torch.Tensor,
                      Ts: float,
                      vehicle_spec: tuple) -> torch.Tensor:
    # parameters
    m, Iz, lf, lr, Cf, Cr, vx_max, vx_min = vehicle_spec
    x, y, vx, vy, phi, omega = torch.unbind(ego_state, dim=-1)
    ax, steer = torch.unbind(action, dim=-1)

    return torch.stack([
        x + Ts * (vx * torch.cos(phi) - vy * torch.sin(phi)),
        y + Ts * (vy * torch.cos(phi) + vx * torch.sin(phi)),
        torch.clamp(vx + Ts * ax, vx_min, vx_max),
        (-(lf * Cf - lr * Cr) * omega + Cf * steer * vx + m * omega * vx * vx - m * vx * vy / Ts) / (Cf + Cr - m * vx / Ts),
        phi + Ts * omega,
        (-Iz * omega * vx / Ts - (lf * Cf - lr * Cr) * vy + lf * Cf * steer * vx) / (
            (lf * lf * Cf + lr * lr * Cr) - Iz * vx / Ts)
    ], dim=-1)
