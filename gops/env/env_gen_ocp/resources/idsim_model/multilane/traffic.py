import numpy as np

# from idsim.envs.env import CrossRoad
from gops.env.env_gen_ocp.pyth_base import Env as CrossRoad
from gops.env.env_gen_ocp.resources.idsim_model.params import ModelConfig


def get_traffic_light_param(env: CrossRoad,
                            model_config: ModelConfig) -> np.ndarray:
    N = model_config.N
    # from vehicle CG to stopline
    if env.engine.context.vehicle.ahead_lane_length != -1:
        ahead_lane_length = env.engine.context.vehicle.ahead_lane_length + \
            env.engine.context.vehicle.length * 0.5
    else:
        ahead_lane_length = env.engine.context.vehicle.ahead_lane_length
    remain_phase_time = env.engine.context.vehicle.remain_phase_time
    in_junction = env.engine.context.vehicle.in_junction
    if ahead_lane_length < model_config.ahead_lane_length_max \
            and ahead_lane_length >= 0.:
        traffic_light = encode_traffic_light(
            env.engine.context.vehicle.traffic_light)
    else:
        traffic_light = encode_traffic_light('g')
    traffic_light_param = np.ones((N+1, 3))
    traffic_light_param[:, 0] = traffic_light * np.ones((N+1))
    traffic_light_param[:, 1] = ahead_lane_length * np.ones((N+1))
    traffic_light_param[:, 2] = in_junction * np.ones((N+1))
    return traffic_light_param


def encode_traffic_light(traffic_light: str) -> np.ndarray:
    if traffic_light in 'Gg':
        return 0
    else:
        return 1
