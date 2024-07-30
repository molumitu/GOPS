from typing import Dict, Union, Tuple
import numpy as np

MAP_ROOT_CROSSROAD = ''
MAP_ROOT_MULTILANE = '/home/toyota/code/idsim-scenarios/idsim-multilane-cross-dense-v20-mix-multi-size-v3/'
pre_horizon = 30
delta_t = 0.1

MAX_STEERING_ANGLE_MULTILANE = 0.065

DELTA_ACC_LB, DELTA_ACC_UB = -2.5 * delta_t, 2.5 * delta_t
REAL_ACC_LB, REAL_ACC_UB = -3.0, 0.8

env_config_param_base = {
    "use_render": False,
    "seed": 1,
    "actuator": "ExternalActuator",
    "scenario_reuse": 4,
    "num_scenarios": 19,
    "detect_range": 60,
    "choose_vehicle_retries": 10,
    "scenario_selector": None,
    "extra_sumo_args": ("--start", "--delay", "200"),
    "warmup_time": 10.0,
    "ignore_traffic_lights": False,
    "incremental_action": True,
    "obs_num_surrounding_vehicles": {
        "passenger": 6,
        "bicycle": 2,
        "pedestrian": 4,
    },
    "ref_v": 15.0,
    "ref_length": 48.0,
    "obs_num_ref_points": 2 * pre_horizon + 1,
    "obs_ref_interval": 0.8,
    "vehicle_spec": (1880.0, 1536.7, 1.13, 1.52, -128915.5, -85943.6, 20.0, 0.0),
    "singleton_mode": "reuse",
    "scenario_root": MAP_ROOT_MULTILANE,
    "max_steps": 1000,
    "use_multiple_path_for_multilane": True,
    "random_ref_probability": 0.01,
    "random_ref_cooldown": 100,
    "no_done_at_collision": True,
    "takeover_bias": True,
    "action_lower_bound": (DELTA_ACC_LB, -MAX_STEERING_ANGLE_MULTILANE * delta_t),
    "action_upper_bound": (DELTA_ACC_UB, MAX_STEERING_ANGLE_MULTILANE * delta_t),
    "real_action_lower_bound": (REAL_ACC_LB, -MAX_STEERING_ANGLE_MULTILANE),
    "real_action_upper_bound": (REAL_ACC_UB, MAX_STEERING_ANGLE_MULTILANE),

    "use_random_acc": True,
    "random_acc_cooldown": (0, 150, 250),  # cooldown for acceleration, deceleration and ref_v, respectively
    "random_acc_prob": (0.0, 0.2),  # probability to accelerate and decelerate, respectively
    "random_acc_range": (0.0, 0.0),  # (m/s^2), used for acceleration (now useless)
    "random_dec_range": (-2.0, -0.5),  # (m/s^2), used for deceleration
}

model_config_base = {
    "N": pre_horizon,
    "sur_obs_padding": "rule",
    "add_boundary_obs": False,
    "full_horizon_sur_obs": False,
    "ahead_lane_length_min": 6.0,
    "ahead_lane_length_max": 60.0,
    "v_discount_in_junction_straight": 0.75,
    "v_discount_in_junction_left_turn": 0.5,
    "v_discount_in_junction_right_turn": 0.375,
    "num_ref_lines": 3,
    "dec_before_junction_green": 0.8,
    "dec_before_junction_red": 1.3,
    "ego_length": 5.0,
    "ego_width": 1.8,
    "safe_dist_incremental": 1.2,

    "num_ref_points": pre_horizon + 1,
    "downsample_ref_point_index": tuple([i for i in range(pre_horizon + 1)]),
    "ego_feat_dim": 7,  # vx, vy, r, last_last_acc, last_last_steer, last_acc, last_steer
    "ego_bound_dim": 2,  # left, right
    "per_sur_state_dim": 6,  # x, y, phi, speed, length, width
    "per_sur_state_withinfo_dim": 7,  # x, y, phi, speed, length, width, mask
    "per_sur_feat_dim": 5,  # x, y, cos(phi), sin(phi), speed
    "per_ref_feat_dim": 5,  # x, y, cos(phi), sin(phi), speed
    "steer_rate_2_min": -0.2,
    "steer_rate_2_max": 0.2,

    "vx_min": 0.0,
    "vx_max": 20.0,
    "vy_min": -4.0,
    "vy_max": 4.0,

    "max_dist_from_ref": 1.8,

     "Q": (
        0.0,
        5.0,
        1000.0,
        5.0,
        0.0,
        100.0,
    ),
    "R": (
        3.0,
        2000.0,
    ),

    "C_acc_rate_1": 1.0,
    "C_steer_rate_1": 100.0,
    "C_steer_rate_2": (0.0, 0.0), # C_steer_rate_2_min, C_steer_rate_2_max
    "C_v": (0., 0., 0., 0.),  # C_vx_min, C_vx_max, C_vy_min, C_vy_max

    "C_lat": 0.0,
    "C_obs": 300.0,
    "C_back": (
        0.5,  # surr is behind ego
        1.0  # surr is in front of ego
    ),
    "C_road": 0.0,
    "ref_v_lane": 15.0,
    "filter_num": 5,
    "track_closest_ref_point": True,
    "real_action_lower": (REAL_ACC_LB, -MAX_STEERING_ANGLE_MULTILANE),
    "real_action_upper": (REAL_ACC_UB, MAX_STEERING_ANGLE_MULTILANE),
    "use_nominal_action": False,
}

env_config_param_multilane = env_config_param_base

model_config_multilane = model_config_base


def get_idsim_env_config(scenario="crossroad") -> Dict:
    if scenario == "crossroad":
        return env_config_param_crossroad
    elif scenario == "multilane":
        return env_config_param_multilane
    else:
        raise NotImplementedError


def get_idsim_model_config(scenario="crossroad") -> Dict:
    if scenario == "crossroad":
        return model_config_crossroad
    elif scenario == "multilane":
        return model_config_multilane
    else:
        raise NotImplementedError


def cal_idsim_obs_scale(
        ego_scale: Union[float, list] = 1.0,
        sur_scale: Union[float, list] = 1.0,
        ref_scale: Union[float, list] = 1.0,
        env_config: Dict = None,
        env_model_config: Dict = None,
):
    ego_dim = env_model_config["ego_feat_dim"]
    sur_dim = env_model_config["per_sur_feat_dim"] + 3  # +3 for length, width, mask
    ref_dim = env_model_config["per_ref_feat_dim"]
    sur_num = int(sum(i for i in env_config["obs_num_surrounding_vehicles"].values()))
    full_horizon_sur_obs = env_model_config["full_horizon_sur_obs"]
    num_ref_points = len(env_model_config["downsample_ref_point_index"])

    if isinstance(ego_scale, float):
        ego_scale = [ego_scale] * ego_dim
    if isinstance(sur_scale, float):
        sur_scale = [sur_scale] * sur_dim
    if isinstance(ref_scale, float):
        ref_scale = [ref_scale] * ref_dim

    assert len(ego_scale) == ego_dim, f"len(ego_scale)={len(ego_scale)}, ego_dim={ego_dim}"
    assert len(sur_scale) == sur_dim, f"len(sur_scale)={len(sur_scale)}, sur_dim={sur_dim}"
    assert len(ref_scale) == ref_dim, f"len(ref_scale)={len(ref_scale)}, ref_dim={ref_dim}"

    obs_scale = []
    obs_scale += ego_scale

    for scale in ref_scale:
        obs_scale += [scale] * num_ref_points

    if full_horizon_sur_obs:
        obs_scale += (sur_scale * sur_num * num_ref_points)
    else:
        obs_scale += sur_scale * sur_num

    obs_scale = np.array(obs_scale, dtype=np.float32)
    return obs_scale


def cal_idsim_pi_paras(
        env_config: Dict = None,
        env_model_config: Dict = None,
):
    ego_dim = env_model_config["ego_feat_dim"]
    sur_dim = env_model_config["per_sur_feat_dim"] + 3  # +3 for length, width, mask
    ref_dim = env_model_config["per_ref_feat_dim"]
    num_ref_points = len(env_model_config["downsample_ref_point_index"])
    num_objs = int(sum(i for i in env_config["obs_num_surrounding_vehicles"].values()))

    pi_paras = {}
    pi_paras["pi_begin"] = ego_dim + ref_dim * num_ref_points
    pi_paras["pi_end"] = pi_paras["pi_begin"] + sur_dim * num_objs
    pi_paras["obj_dim"] = sur_dim
    pi_paras["output_dim"] = sur_dim * num_objs + 1
    return pi_paras

