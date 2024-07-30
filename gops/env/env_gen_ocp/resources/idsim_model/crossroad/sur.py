import numpy as np
import torch
from typing import List, Dict

# from idsim.envs.env import CrossRoad
from gops.env.env_gen_ocp.pyth_base import Env as CrossRoad
from gops.env.env_gen_ocp.resources.idsim_model.utils.numpy_utils import pad_with_mask
from gops.env.env_gen_ocp.resources.idsim_model.params import ModelConfig, sur_noise_dict


def remove_irrelevant_vehicle(surrounding,
                              vehicle,
                              network):
    # Keep vehicles in:
    # 1. the same route
    # 2. upcoming/current junction (junction's internal edges)
    # 3. upcoming/current junction's incoming edges
    if vehicle.edge == vehicle.route[-1] and not vehicle.in_junction:
        # The second condition excludes an edge case where the vehicle drives
        # back to the junction after leaving it, same as in navigation module.
        return [s for s in surrounding if s.road_id == vehicle.edge]

    if vehicle.in_junction:
        junction_id, _ = network.get_junction_by_edge_hint(
            vehicle.ego_polygon, vehicle.edge)
    else:
        junction_id = network.get_upcoming_junction(vehicle.edge)
    incoming_edges = network.get_junction_incoming_edges(junction_id)
    allowed_edges = incoming_edges + vehicle.route
    return [s for s in surrounding if (s.road_id in allowed_edges or network.is_edge_internal(s.road_id))]


def get_sur_state(env: CrossRoad,
                  model_config: ModelConfig,
                  rng: np.random.Generator) -> np.ndarray:
    per_sur_state_dim = model_config.per_sur_state_dim
    num_veh = env.config.obs_num_surrounding_vehicles['passenger']
    num_bike = env.config.obs_num_surrounding_vehicles['bicycle']
    num_ped = env.config.obs_num_surrounding_vehicles['pedestrian']
    sur_info = env.engine.context.vehicle.surrounding_veh_info
    sur_info = [s for s in sur_info if s.rel_x > 0.]

    # split the surrounding vehicles into three types
    sur_veh_info = [s for s in sur_info if s.type.startswith('v')]
    sur_bike_info = [s for s in sur_info if s.type == 'b1']
    sur_ped_info = [s for s in sur_info if s.type == 'DEFAULT_PEDTYPE']

    # only choose the M closest vehicles
    sur_veh_state = construct_state(
        model_config, sur_veh_info, num_veh, per_sur_state_dim, sur_noise_dict['passenger'], rng)
    sur_bike_state = construct_state(
        model_config, sur_bike_info, num_bike, per_sur_state_dim, sur_noise_dict['bicycle'], rng)
    sur_ped_state = construct_state(
        model_config, sur_ped_info, num_ped, per_sur_state_dim, sur_noise_dict['pedestrian'], rng)
    sur_state_withinfo = np.concatenate(
        (sur_veh_state, sur_bike_state, sur_ped_state), axis=0)
    return sur_state_withinfo


def get_sur_param(env: CrossRoad,
                  model_config: ModelConfig,
                  rng: np.random.Generator) -> np.ndarray:
    # (M, per_sur_state_withinfo_dim)
    sur_state_withinfo = get_sur_state(env, model_config, rng)
    M = sur_state_withinfo.shape[0]
    N = model_config.N
    per_sur_state_withinfo_dim = model_config.per_sur_state_withinfo_dim
    # predict the future
    sur_param = np.zeros(
        (2*N+1, M, per_sur_state_withinfo_dim), dtype=np.float32)
    sur_param[0] = sur_state_withinfo
    for i in range(1, 2*N+1):
        sur_param[i] = sur_predict_model(sur_param[i-1], env.config.dt)
    return sur_param


def construct_state(model_config: ModelConfig,
                    sur_info,
                    M: int,
                    per_sur_state_dim: int,
                    noise_dict: Dict[str, np.ndarray],
                    rng: np.random.Generator) -> np.ndarray:
    sur_info.sort(key='distance')
    sur_state = [(s.x, s.y, s.phi, s.speed, s.length, s.width)
                 for s in sur_info[:M]]
    # sur_state = [(s.x, s.y, s.phi, s.speed,
    #               max(s.length, model_config.min_dist_sur_length),
    #               max(s.width, model_config.min_dist_sur_width)) for s in sur_info[:M]]
    # rng.shuffle(sur_state)
    sur_state = np.array(sur_state, dtype=np.float32).reshape(
        (-1, per_sur_state_dim))
    # add noise
    noise_mean, noise_std, noise_upper_bound, noise_lower_bound = noise_dict[
        'mean'], noise_dict['std'], noise_dict['upper_bound'], noise_dict['lower_bound']
    noise = rng.normal(noise_mean, noise_std, size=sur_state.shape)
    noise = np.clip(noise, noise_lower_bound, noise_upper_bound)
    sur_state += noise

    # pad with zeros & add mask
    sur_state_withinfo = pad_with_mask(sur_state, M, separate_mask=False)
    return sur_state_withinfo


def sur_predict_model(sur_state: np.ndarray, Ts: float) -> np.ndarray:
    x, y, phi, speed, length, width, mask = sur_state.T

    return np.array([
        x + Ts * speed * np.cos(phi),
        y + Ts * speed * np.sin(phi),
        phi,
        speed,
        length,
        width,
        mask
    ]).T


def get_sur_param_requires_grad(sur_param: torch.Tensor,
                                model_config: ModelConfig) -> torch.Tensor:
    N = model_config.N
    sur_state = sur_param[:, 0]
    sur_state.unsqueeze_(1)
    sur_state.requires_grad = True

    # predict the future
    sur_param_requires_grad = []
    sur_param_requires_grad.append(sur_state)
    for virtual_timestep in range(N):
        sur_param_requires_grad.append(sur_predict_model_requires_grad(
            sur_param_requires_grad[virtual_timestep]))
    sur_param_requires_grad = torch.concat(sur_param_requires_grad, dim=1)
    return sur_param_requires_grad


def sur_predict_model_requires_grad(sur_state: torch.Tensor) -> torch.Tensor:
    Ts = 0.1
    x, y, phi, speed, mask = sur_state.unbind(-1)

    return torch.stack([
        x + Ts * speed * torch.cos(phi),
        y + Ts * speed * torch.sin(phi),
        phi,
        speed,
        mask
    ], dim=-1)
