import numpy as np
from typing import List, Tuple
from omegaconf import OmegaConf
from dataclasses import dataclass
@dataclass(frozen=False)
class ModelConfig():
    N: int = 30
    num_ref_points: int = 31
    full_horizon_sur_obs: bool = False
    sur_obs_padding: str = "zero" # "zero or "rule"
    ego_feat_dim: int = 7 # vx, vy, r, last_last_acc, last_last_steer, last_acc, last_steer
    add_boundary_obs: bool = False
    ego_bound_dim: int = 2 # left, right
    per_sur_state_dim: int = 6 # x, y, phi, speed, length, width
    per_sur_state_withinfo_dim: int = 7 # x, y, phi, speed, length, width, mask
    per_sur_feat_dim: int = 5 # x, y, cos(phi), sin(phi), speed
    per_ref_feat_dim: int = 5 # x, y, cos(phi), sin(phi), speed
    ref_v_lane: float = 8.0 # default for urban
    v_discount_in_junction_straight: float = 0.75
    v_discount_in_junction_left_turn: float = 0.5
    v_discount_in_junction_right_turn: float = 0.5
    num_ref_lines: int = 3
    downsample_ref_point_index: Tuple[int] = tuple([i for i in range(31)])
    filter_num: int = 0  # only for extra filter
    ahead_lane_length_min: float = 6.0
    ahead_lane_length_max: float = 60.0
    dec_before_junction: float = 0.8
    dec_before_junction_green: float = 0.8
    dec_before_junction_red: float = 1.3
    ego_length: float = 5.0
    ego_width: float = 1.8
    padding_veh_shape: Tuple[float, float] = (5.0, 1.8)
    padding_bike_shape: Tuple[float, float] = (0.0, 0.0)  # TODO: need to be modified
    padding_ped_shape: Tuple[float, float] = (0.0, 0.0)
    safe_dist_incremental: float = 1.2 # same with IDC problem
    min_dist_sur_length: float = 1.8
    min_dist_sur_width: float = 1.8

    real_action_upper: Tuple[float] = (0.8, 0.065) # [acc, steer]
    real_action_lower: Tuple[float] = (-3.0, -0.065) # [acc, steer]

    steer_rate_2_min: float = -0.2
    steer_rate_2_max: float = 0.2

    vx_min: float = 0.0
    vx_max: float = 20.0 # (m/s)
    vy_min: float = -4.0
    vy_max: float = 4.0

    max_dist_from_ref: float = 1.8 # (self added)

    Q: Tuple[float] = (0.4, 0.4, 500., 1., 1., 300.0)
    R: Tuple[float] = (1.0, 20.0,)

    track_closest_ref_point: bool = False
    use_nominal_action: bool = False
    ref_v_slow_focus: float = 0. # focus more on low speed tracking when ref_v < ref_v_slow_focus
    Q_slow_incre: Tuple[float] = (0., 0., 0., 0., 0., 0.) # when ref_v < ref_v_slow_focus, increment Q
    R_slow_incre: Tuple[float] = (0., 0.) # when ref_v < ref_v_slow_focus, increment R
    clear_nonessential_cost_safe: bool = False # clear the safe cost of some nonessential obstacles

    C_acc_rate_1: float = 0.0
    C_steer_rate_1: float = 0.0
    C_steer_rate_2: Tuple[float] = (100., 100.) # C_steer_rate_2_min, C_steer_rate_2_max
    C_v: Tuple[float] = (100., 100., 100., 100.) # C_vx_min, C_vx_max, C_vy_min, C_vy_max

    gamma: float = 0.99 # should be equal to discount factor of algorithm
    lambda_c: float = 0.99  # discount of lat penalty
    lambda_p: float = 0.99  # discount of lon penalty
    C_lat: float = 3.0
    C_obs: float = 300.0
    C_back: Tuple[float] = (0.1, 1.0)
    C_road: float = 300.0

    reward_scale: float = 0.01
    reward_comps: Tuple[str] = ()

    @staticmethod
    def from_partial_dict(partial_dict) -> "ModelConfig":
        base = OmegaConf.structured(ModelConfig)
        merged = OmegaConf.merge(base, partial_dict)
        return OmegaConf.to_object(merged)

model_config = {
    "N": 30,
    "num_ref_points": 31,
    "ego_feat_dim": 7, # vx, vy, r, last_last_acc, last_last_steer, last_acc, last_steer
    "add_boundary_obs": False,
    "ego_bound_dim": 2, # left, right
    "per_sur_state_dim": 6, # x, y, phi, speed, length, width
    "per_sur_state_withinfo_dim": 7, # x, y, phi, speed, length, width, mask
    "per_sur_feat_dim": 5, # x, y, cos(phi), sin(phi), speed
    "per_ref_feat_dim": 5, # x, y, cos(phi), sin(phi), speed
    "filter_num": 0,  # only for extra filter
}

noise_params = {
    "passenger": {
        "std_x": 0.004340,
        "std_y": 0.006041,
        "std_vx": 0.075129,
        "std_phi": 0.005316,
        "std_length": 0.3,
        "std_width": 0.15,
        "upper_x": 0.012575,
        "upper_y": 0.018247,
        "upper_vx": 0.225307,
        "upper_phi": 0.016026,
        "upper_length": 1.0,
        "upper_width": 0.4,
        "lower_x": -0.013465,
        "lower_y": -0.017999,
        "lower_vx": -0.225467,
        "lower_phi": -0.015870,
        "lower_length": -0.5,
        "lower_width": -0.2,
    },
    "bicycle": {
        "std_x": 0.001753,
        "std_y": 0.001953,
        "std_vx": 0.026401,
        "std_phi": 0.002167,
        "std_length": 0.3/5.0,
        "std_width": 0.15/5.0,
        "upper_x": 0.005271,
        "upper_y": 0.005924,
        "upper_vx": 0.080495,
        "upper_phi": 0.006433,
        "upper_length": 1.0/5.0,
        "upper_width": 0.4/5.0,
        "lower_x": -0.005247,
        "lower_y": -0.005794,
        "lower_vx": -0.077911,
        "lower_phi": -0.006569,
        "lower_length": -0.5/5.0,
        "lower_width": -0.2/5.0,
    },
    "pedestrian": {
        "std_x": 0.004531,
        "std_y": 0.004795,
        "std_vx": 0.065836,
        "std_phi": 0.036477,
        "std_length": 0.05,
        "std_width": 0.05,
        "upper_x": 0.013559,
        "upper_y": 0.014318,
        "upper_vx": 0.181510,
        "upper_phi": 0.109218,
        "upper_length": 0.15,
        "upper_width": 0.15,
        "lower_x": -0.013627,
        "lower_y": -0.014452,
        "lower_vx": -0.213506,
        "lower_phi": -0.109644,
        "lower_length": 0.0,
        "lower_width": 0.0,
    },
}

ego_mean_array = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
ego_std_array = np.array([noise_params["passenger"]["std_x"],
                          noise_params["passenger"]["std_y"],
                          noise_params["passenger"]["std_vx"],
                          0.0, noise_params["passenger"]["std_phi"],
                          0.0], dtype=np.float32)
ego_upper_bound_array = np.array([noise_params["passenger"]["upper_x"],
                                  noise_params["passenger"]["upper_y"],
                                  noise_params["passenger"]["upper_vx"],
                                  0.0, noise_params["passenger"]["upper_phi"],
                                  0.0], dtype=np.float32)
ego_lower_bound_array = np.array([noise_params["passenger"]["lower_x"],
                                  noise_params["passenger"]["lower_y"],
                                  noise_params["passenger"]["lower_vx"],
                                  0.0, noise_params["passenger"]["lower_phi"],
                                  0.0], dtype=np.float32)
# ## zeros
ego_mean_array = np.zeros_like(ego_mean_array)
ego_std_array = np.zeros_like(ego_std_array)
ego_upper_bound_array = np.zeros_like(ego_upper_bound_array)
ego_lower_bound_array = np.zeros_like(ego_lower_bound_array)

sur_passenger_mean_array = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
sur_passenger_std_array = np.array([noise_params["passenger"]["std_x"],
                                    noise_params["passenger"]["std_y"],
                                    noise_params["passenger"]["std_phi"],
                                    noise_params["passenger"]["std_vx"],
                                    noise_params["passenger"]["std_length"],
                                    noise_params["passenger"]["std_width"]], dtype=np.float32)
sur_passenger_upper_bound_array = np.array([noise_params["passenger"]["upper_x"],
                                            noise_params["passenger"]["upper_y"],
                                            noise_params["passenger"]["upper_phi"],
                                            noise_params["passenger"]["upper_vx"],
                                            noise_params["passenger"]["upper_length"],
                                            noise_params["passenger"]["upper_width"]], dtype=np.float32)
sur_passenger_lower_bound_array = np.array([noise_params["passenger"]["lower_x"],
                                            noise_params["passenger"]["lower_y"],
                                            noise_params["passenger"]["lower_phi"],
                                            noise_params["passenger"]["lower_vx"],
                                            noise_params["passenger"]["lower_length"],
                                            noise_params["passenger"]["lower_width"]], dtype=np.float32)

sur_bicycle_mean_array = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
sur_bicycle_std_array = np.array([noise_params["bicycle"]["std_x"],
                                  noise_params["bicycle"]["std_y"],
                                  noise_params["bicycle"]["std_phi"],
                                  noise_params["bicycle"]["std_vx"],
                                  noise_params["bicycle"]["std_length"],
                                  noise_params["bicycle"]["std_width"]], dtype=np.float32)
sur_bicycle_upper_bound_array = np.array([noise_params["bicycle"]["upper_x"],
                                          noise_params["bicycle"]["upper_y"],
                                          noise_params["bicycle"]["upper_phi"],
                                          noise_params["bicycle"]["upper_vx"],
                                          noise_params["bicycle"]["upper_length"],
                                          noise_params["bicycle"]["upper_width"]], dtype=np.float32)
sur_bicycle_lower_bound_array = np.array([noise_params["bicycle"]["lower_x"],
                                          noise_params["bicycle"]["lower_y"],
                                          noise_params["bicycle"]["lower_phi"],
                                          noise_params["bicycle"]["lower_vx"],
                                          noise_params["bicycle"]["lower_length"],
                                          noise_params["bicycle"]["lower_width"]], dtype=np.float32)

sur_pedestrian_mean_array = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
sur_pedestrian_std_array = np.array([noise_params["pedestrian"]["std_x"],
                                     noise_params["pedestrian"]["std_y"],
                                     noise_params["pedestrian"]["std_phi"],
                                     noise_params["pedestrian"]["std_vx"],
                                     noise_params["pedestrian"]["std_length"],
                                     noise_params["pedestrian"]["std_width"]], dtype=np.float32)
sur_pedestrian_upper_bound_array = np.array([noise_params["pedestrian"]["upper_x"],
                                             noise_params["pedestrian"]["upper_y"],
                                             noise_params["pedestrian"]["upper_phi"],
                                             noise_params["pedestrian"]["upper_vx"],
                                             noise_params["pedestrian"]["upper_length"],
                                             noise_params["pedestrian"]["upper_width"]], dtype=np.float32)
sur_pedestrian_lower_bound_array = np.array([noise_params["pedestrian"]["lower_x"],
                                             noise_params["pedestrian"]["lower_y"],
                                             noise_params["pedestrian"]["lower_phi"],
                                             noise_params["pedestrian"]["lower_vx"],
                                             noise_params["pedestrian"]["lower_length"],
                                             noise_params["pedestrian"]["lower_width"]], dtype=np.float32)

sur_noise_dict = {
    "passenger": {
        "mean": sur_passenger_mean_array,
        "std": sur_passenger_std_array,
        "upper_bound": sur_passenger_upper_bound_array,
        "lower_bound": sur_passenger_lower_bound_array,
    },
    "bicycle": {
        "mean": sur_bicycle_mean_array,
        "std": sur_bicycle_std_array,
        "upper_bound": sur_bicycle_upper_bound_array,
        "lower_bound": sur_bicycle_lower_bound_array,
    },
    "pedestrian": {
        "mean": sur_pedestrian_mean_array,
        "std": sur_pedestrian_std_array,
        "upper_bound": sur_pedestrian_upper_bound_array,
        "lower_bound": sur_pedestrian_lower_bound_array,
    },
}

# zeros
# sur_noise_dict = {
#     "passenger": {
#         "mean": sur_passenger_mean_array,
#         "std": np.zeros_like(sur_passenger_std_array),
#         "upper_bound": np.zeros_like(sur_passenger_upper_bound_array),
#         "lower_bound": np.zeros_like(sur_passenger_lower_bound_array),
#     },
#     "bicycle": {
#         "mean": sur_bicycle_mean_array,
#         "std": np.zeros_like(sur_bicycle_std_array),
#         "upper_bound": np.zeros_like(sur_bicycle_upper_bound_array),
#         "lower_bound": np.zeros_like(sur_bicycle_lower_bound_array),
#     },
#     "pedestrian": {
#         "mean": sur_pedestrian_mean_array,
#         "std": np.zeros_like(sur_pedestrian_std_array),
#         "upper_bound": np.zeros_like(sur_pedestrian_upper_bound_array),
#         "lower_bound": np.zeros_like(sur_pedestrian_lower_bound_array),
#     },
# }
