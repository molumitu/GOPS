from typing import Dict, Union, Tuple
import numpy as np

MAP_ROOT_CROSSROAD = ''
MAP_ROOT_MULTILANE = '/home/toyota/code/idsim-scenarios/idsim-multilane-cross-dense-v20-mix-multi-size/'
pre_horizon = 30
delta_t = 0.1

env_config_param_base = {
    "use_render": False,
    "seed": 1,
    "actuator": "ExternalActuator",
    "scenario_reuse": 4,
    "num_scenarios": 19,
    "detect_range": 60,
    "choose_vehicle_retries": 10,
    "choose_vehicle_step_time": 10,
    "scenario_root": MAP_ROOT_CROSSROAD,
    "scenario_selector": None,
    "direction_selector": None,
    "extra_sumo_args": ("--start", "--delay", "200"),
    "warmup_time": 50.0,
    "max_steps": 200,
    "random_ref_v": True,
    "ref_v_range": (0, 10.0),
    "nonimal_acc": True,
    "ignore_traffic_lights": False,
    "no_done_at_collision": False, 
    "ignore_surrounding": False,
    "ignore_opposite_direction": True,
    "penalize_collision": True,
    "incremental_action": True,
    "action_lower_bound": (-4.0 * delta_t, -0.25 * delta_t),
    "action_upper_bound": (2.5 * delta_t, 0.25 * delta_t),
    "real_action_lower_bound": (-3.0, -0.571),
    "real_action_upper_bound": (0.8, 0.571),
    "obs_num_surrounding_vehicles": {
        "passenger": 8,
        "bicycle": 2,
        "pedestrian": 2,
    },
    "ref_v": 10.0,
    "ref_length": 48.0,
    "obs_num_ref_points": 2 * pre_horizon + 1,
    "obs_ref_interval": 0.8,
    # "vehicle_spec": (1880.0, 1536.7, 1.13, 1.52, -128915.5, -85943.6, 20.0, 0.0),
    "vehicle_spec": (1880.0, 1536.7, 1.22, 1.70, -128915.5, -85943.6, 20.0, 0.0),
    "singleton_mode": "reuse",
    "random_ref_probability": 0.01,
    "use_multiple_path_for_multilane": True,
    "random_ref_cooldown":  80,

    "takeover_bias": True, 
    "takeover_bias_prob": 1.0,
    "takeover_bias_x": (0.0, 0.1),
    "takeover_bias_y": (0.0, 0.1),
    "takeover_bias_phi": (0.0, 0.05),
    "takeover_bias_vx": (0.0, 0.0),
    "takeover_bias_ax": (0.0, 0.0),
    "takeover_bias_steer": (0.0, 0.0),
    "minimum_clearance_when_takeover":-1,
    # model free reward config
    "punish_sur_mode": "max",
    "enable_slow_reward": True,
    "R_step": 10.0,
    "P_lat": 12.0,
    "P_long": 10.0,
    "P_phi": 3.0,
    "P_yaw": 2.0,
    "P_front": 10.0,
    "P_side": 10.0,
    "P_space": 10.0,
    "P_rear": 10.0,
    "P_steer": 0.1,
    "P_acc": 0.1,
    "P_delta_steer": 0.15,
    "P_jerk": 0.1,
    "P_done": 200.0,
    "P_boundary": 0,
    "safety_lat_margin_front": 1.2,
    "safety_long_margin_front": 0.0,
    "safety_long_margin_side": 0.0,
    "front_dist_thd": 50.0,
    "space_dist_thd": 12.0,
    "rel_v_thd": 1.0,
    "rel_v_rear_thd": 3.0,
    "time_dist": 1,
}

model_config_base = {
    "N": pre_horizon,
    "sur_obs_padding": "rule",
    "add_boundary_obs": False,
    "full_horizon_sur_obs": False,
    "ahead_lane_length_min": 6.0,
    "ahead_lane_length_max": 60.0,
    "v_discount_in_junction_straight": 0.75,
    "v_discount_in_junction_left_turn": 0.3,
    "v_discount_in_junction_right_turn": 0.3,
    "num_ref_lines": 3,
    "dec_before_junction_green": 0.8,
    "dec_before_junction_red": 1.3,
    "ego_length": 5.0,
    "ego_width": 1.8,
    "safe_dist_incremental": 1.2,
    "downsample_ref_point_index": (0, 1, 10, 30),

    "num_ref_points": pre_horizon + 1,
    "ego_feat_dim": 7,  # vx, vy, r, last_last_acc, last_last_steer, last_acc, last_steer
    "ego_bound_dim": 2,  # left, right
    "per_sur_state_dim": 6,  # x, y, phi, speed, length, width
    "per_sur_state_withinfo_dim": 7,  # x, y, phi, speed, length, width, mask
    "per_sur_feat_dim": 5,  # x, y, cos(phi), sin(phi), speed
    "per_ref_feat_dim": 5,  # x, y, cos(phi), sin(phi), speed
    "real_action_upper": (0.8, 0.571),
    "real_action_lower": (-3.0, -0.571),
    "steer_rate_2_min": -0.2,
    "steer_rate_2_max": 0.2,

    "vx_min": 0.0,
    "vx_max": 20.0,
    "vy_min": -4.0,
    "vy_max": 4.0,

    "max_dist_from_ref": 1.8,

    "Q": (
        0.4,
        0.4,
        500.0,
        1.0,
        2.0,
        300.0,
    ),
    "R": (
        1.0,
        20.0,
    ),

    "C_acc_rate_1": 0.0,
    "C_steer_rate_1": 10.0,
    "C_steer_rate_2": (10.0, 10.0), # C_steer_rate_2_min, C_steer_rate_2_max
    "C_v": (100., 100., 100., 100.), # C_vx_min, C_vx_max, C_vy_min, C_vy_max

    "gamma": 1.0,  # should equal to discount factor of algorithm
    "lambda_c": 0.99,  # discount of lat penalty
    "lambda_p": 0.99,  # discount of lon penalty
    "C_lat": 3.0,
    "C_obs": 300.0,
    "C_back": (
        0.1,  # surr is behind ego
        1.0  # surr is in front of ego
    ),
    "C_road": 300.0,
    "ref_v_lane": 10.0,
    "filter_num": 5
}

env_config_param_crossroad = env_config_param_base
model_config_crossroad = model_config_base

env_config_param_multilane = {
    **env_config_param_base,
    "scenario_root": MAP_ROOT_MULTILANE,
    "action_lower_bound": (-2.5 * delta_t, -0.065 * delta_t),
    "action_upper_bound": (2.5 * delta_t, 0.065 * delta_t),
    "real_action_lower_bound": (-3.0, -0.065),
    "real_action_upper_bound": (0.8, 0.065),
    "use_random_acc": True,
    "random_acc_cooldown": (30, 50, 50), # cooldown for acceleration, deceleration and ref_v, respectively
    "random_acc_prob": (0.1, 0.1), # probability to accelerate and decelerate, respectively
    "random_acc_range": (0.2, 0.8), # (m/s^2), used for acceleration
    "random_dec_range": (-1.5, -0.5), # (m/s^2), used for deceleration
}

model_config_multilane = {
    **model_config_base,
    "real_action_lower": (-3.0, -0.065),
    "real_action_upper": (0.8, 0.065),
    "Q": (
        0.,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    "R": (
        0.0,
        0.0,
    ),
    # "reward_comps": ( 
    #     "env_pun2front",
    #     "env_pun2side",
    #     "env_pun2space",
    #     "env_pun2rear",
    #     "env_reward_vel_long",
    #     "env_reward_yaw_rate",
    #     "env_reward_dist_lat",
    #     "env_reward_head_ang",
    #     "env_reward_steering",
    #     "env_reward_acc_long",
    #     "env_reward_delta_steer",
    #     "env_reward_jerk",
    # )
    "reward_comps": ( 
        "env_pun2front",
        "env_pun2side",
        "env_pun2space",
        "env_pun2rear",
        "env_reward_vel_long",
        "env_reward_yaw_rate",
        "env_reward_dist_lat",
        "env_reward_head_ang",
        "env_reward_steering",
        "env_reward_acc_long",
        "env_reward_delta_steer",
        "env_reward_jerk",
    )
}


def get_idsim_env_config(scenario="multilane") -> Dict:
    if scenario == "crossroad":
        return env_config_param_crossroad
    elif scenario == "multilane":
        return env_config_param_multilane
    else:
        raise NotImplementedError


def get_idsim_model_config(scenario="multilane") -> Dict:
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
        bound_scale: Union[float, list] = 1.0,
        env_config: Dict = None,
        env_model_config: Dict = None,
):
    ego_dim = env_model_config["ego_feat_dim"]
    sur_dim = env_model_config["per_sur_feat_dim"] + 3 # +3 for length, width, mask
    ref_dim = env_model_config["per_ref_feat_dim"]
    bound_dim = env_model_config["ego_bound_dim"]
    sur_num = int(sum(i for i in env_config["obs_num_surrounding_vehicles"].values()))
    full_horizon_sur_obs = env_model_config["full_horizon_sur_obs"]
    num_ref_points = len(env_model_config["downsample_ref_point_index"]) 

    if isinstance (ego_scale, float):
        ego_scale = [ego_scale] * ego_dim
    if isinstance (sur_scale, float):
        sur_scale = [sur_scale] * sur_dim
    if isinstance (ref_scale, float):
        ref_scale = [ref_scale] * ref_dim
    if isinstance (bound_scale, float):
        bound_scale = [bound_scale] * bound_dim

    assert len(ego_scale) == ego_dim, f"len(ego_scale)={len(ego_scale)}, ego_dim={ego_dim}"
    assert len(sur_scale) == sur_dim, f"len(sur_scale)={len(sur_scale)}, sur_dim={sur_dim}"
    assert len(ref_scale) == ref_dim, f"len(ref_scale)={len(ref_scale)}, ref_dim={ref_dim}"
    assert len(bound_scale) == bound_dim, f"len(boundary_scale)={len(bound_scale)}, bound_dim={bound_dim}"
    
    obs_scale = []
    obs_scale += ego_scale

    for scale in ref_scale:
        obs_scale += [scale] * num_ref_points

    if full_horizon_sur_obs:
        obs_scale += (sur_scale * sur_num * num_ref_points)
    else:
        obs_scale += sur_scale * sur_num
    if env_model_config["add_boundary_obs"]:
        obs_scale += bound_scale

    obs_scale = np.array(obs_scale, dtype=np.float32)
    return obs_scale

def cal_idsim_pi_paras(
        env_config: Dict = None,
        env_model_config: Dict = None,
):
    ego_dim = env_model_config["ego_feat_dim"]
    sur_dim = env_model_config["per_sur_feat_dim"] + 3 # +3 for length, width, mask
    ref_dim = env_model_config["per_ref_feat_dim"]
    num_ref_points = len(env_model_config["downsample_ref_point_index"]) 
    num_objs = int(sum(i for i in env_config["obs_num_surrounding_vehicles"].values()))

    pi_paras = {}
    pi_paras["pi_begin"] = ego_dim + ref_dim*num_ref_points
    pi_paras["pi_end"] = pi_paras["pi_begin"] + sur_dim*num_objs 
    pi_paras["obj_dim"] = sur_dim 
    pi_paras["output_dim"] = sur_dim*num_objs + 1
    return pi_paras

